import networkx as nx
import argparse, os, re
from collections import deque

INPUT_BENCH = None
GML_DUMP_LOC = None

nodeType = {
    "PI": 0,
    "PO": 1,
    "Internal": 2,
    "DFF": 3,
}

edgeType = {
    "BUFF": 0,
    "NOT": 1,
}

def setGlobalAndEnvironmentVars(cmdArgs):
    benchFile = cmdArgs.bench
    gmlDumpLoc = cmdArgs.gml

    if not (os.path.exists(benchFile)) and os.path.exists(gmlDumpLoc):
        print("Paths are invalid. Please rerun")
        exit(1)

    global INPUT_BENCH, GML_DUMP_LOC
    INPUT_BENCH = benchFile
    GML_DUMP_LOC = gmlDumpLoc

def processANDAssignments(inputs, output, idxCounter, poList, nodeNameIDMapping, singleGateInputIOMapping, AIG_DAG):
    nType = nodeType["Internal"]
    nodeAttributedDict = {
        "node_id": output,
        "node_type": nType,
        "num_inverted_predecessors": 0,
        "fanouts": 0,
        "depth": 0,
        "is_sequential": False
    }
    AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
    nodeNameIDMapping[output] = idxCounter
    numInvertedPredecessors = 0
    for inp in inputs:
        if not (inp in nodeNameIDMapping.keys()):
            srcIdx = nodeNameIDMapping[singleGateInputIOMapping[inp]]
            eType = edgeType["NOT"]
            numInvertedPredecessors += 1
        else:
            srcIdx = nodeNameIDMapping[inp]
            eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
    AIG_DAG.nodes[idxCounter]["num_inverted_predecessors"] = numInvertedPredecessors

    if output in poList:
        nType = nodeType["PO"]
        nodeAttributedDict = {
            "node_id": output + "_buff",
            "node_type": nType,
            "num_inverted_predecessors": 0,
            "fanouts": 0,
            "depth": 0,
            "is_sequential": False
        }
        AIG_DAG.add_nodes_from([(idxCounter + 1, nodeAttributedDict)])
        nodeNameIDMapping[output + "_buff"] = idxCounter + 1
        srcIdx = idxCounter
        eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter + 1, srcIdx, edge_type=eType)

def processDFFAssignments(data_input, output, idxCounter, poList, nodeNameIDMapping, singleGateInputIOMapping, AIG_DAG):
    nType = nodeType["DFF"]
    nodeAttributedDict = {
        "node_id": output,
        "node_type": nType,
        "num_inverted_predecessors": 0,
        "fanouts": 0,
        "depth": 0,
        "is_sequential": True
    }
    AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
    nodeNameIDMapping[output] = idxCounter
    
    if data_input in nodeNameIDMapping:
        srcIdx = nodeNameIDMapping[data_input]
        eType = edgeType["BUFF"]
    elif data_input in singleGateInputIOMapping:
        srcIdx = nodeNameIDMapping[singleGateInputIOMapping[data_input]]
        eType = edgeType["NOT"]
    else:
        new_idx = max(nodeNameIDMapping.values()) + 1 if nodeNameIDMapping else 0
        AIG_DAG.add_node(new_idx, 
                        node_id=data_input,
                        node_type=nodeType["PI"],
                        num_inverted_predecessors=0,
                        fanouts=0,
                        depth=0,
                        is_sequential=False)
        nodeNameIDMapping[data_input] = new_idx
        srcIdx = new_idx
        eType = edgeType["BUFF"]
    
    AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)

    if output in poList:
        po_attrib = {
            "node_id": output + "_buff",
            "node_type": nodeType["PO"],
            "num_inverted_predecessors": 0,
            "fanouts": 0,
            "depth": 0,
            "is_sequential": False
        }
        AIG_DAG.add_node(idxCounter + 1, **po_attrib)
        nodeNameIDMapping[output + "_buff"] = idxCounter + 1
        AIG_DAG.add_edge(idxCounter + 1, idxCounter, edge_type=edgeType["BUFF"])

def compute_node_depths(AIG_DAG):
    for node in AIG_DAG.nodes:
        AIG_DAG.nodes[node]["depth"] = 0

    temp_dag = AIG_DAG.copy()
    edges_to_remove = []
    for u, v, data in temp_dag.edges(data=True):
        if temp_dag.nodes[u]["node_type"] == nodeType["DFF"]:
            edges_to_remove.append((u, v))
    temp_dag.remove_edges_from(edges_to_remove)

    try:
        topo_order = list(nx.topological_sort(temp_dag))
        
        for node in topo_order:
            predecessors = list(AIG_DAG.predecessors(node))
            if predecessors:
                max_pred_depth = max(AIG_DAG.nodes[p]["depth"] for p in predecessors)
                AIG_DAG.nodes[node]["depth"] = max_pred_depth + 1
    except nx.NetworkXUnfeasible:
        print("Warning: Graph contains cycles, using fallback depth calculation")
        for node in nx.algorithms.dag.lexicographical_topological_sort(temp_dag):
            predecessors = list(AIG_DAG.predecessors(node))
            if predecessors:
                max_pred_depth = max(AIG_DAG.nodes[p]["depth"] for p in predecessors)
                AIG_DAG.nodes[node]["depth"] = max_pred_depth + 1

def parseAIGBenchAndCreateNetworkXGraph():
    nodeNameIDMapping = {}
    singleInputgateIOMapping = {}
    poList = []
    benchFile = open(INPUT_BENCH, 'r+')
    benchFileLines = benchFile.readlines()
    benchFile.close()
    AIG_DAG = nx.DiGraph()
    idxCounter = 0
    
    for line in benchFileLines:
        if len(line) == 0 or line.__contains__("ABC"):
            continue
        elif line.__contains__("vdd"):
            line = line.replace(" ", "")
            pi = re.search("(.*?)=", str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0,
                "fanouts": 0,
                "depth": 0,
                "is_sequential": False
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter += 1
        elif line.__contains__("INPUT"):
            line = line.replace(" ", "")
            pi = re.search("INPUT\((.*?)\)", str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0,
                "fanouts": 0,
                "depth": 0,
                "is_sequential": False
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter += 1
        elif line.__contains__("OUTPUT"):
            line = line.replace(" ", "")
            po = re.search("OUTPUT\((.*?)\)", str(line)).group(1)
            poList.append(po)
        elif line.__contains__("AND"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            input1 = re.search("AND\((.*?),", str(line)).group(1)
            input2 = re.search(",(.*?)\)", str(line)).group(1)
            processANDAssignments([input1, input2], output, idxCounter, poList, nodeNameIDMapping, singleInputgateIOMapping, AIG_DAG)
            if output in poList:
                idxCounter += 1
            idxCounter += 1
        elif line.__contains__("NOT"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("NOT\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            if output in poList:
                nodeAttributedDict = {
                    "node_id": output + "_inv",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": 1,
                    "fanouts": 0,
                    "depth": 0,
                    "is_sequential": False
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output + "_inv"] = idxCounter
                srcIdx = nodeNameIDMapping[inputPin]
                eType = edgeType["NOT"]
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        elif line.__contains__("BUFF"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("BUFF\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            numInvertedPredecessors = 0
            if output in poList:
                if inputPin in nodeNameIDMapping.keys():
                    srcIdx = nodeNameIDMapping[inputPin]
                    eType = edgeType["BUFF"]
                else:
                    srcIdx = nodeNameIDMapping[singleInputgateIOMapping[inputPin]]
                    eType = edgeType["NOT"]
                    numInvertedPredecessors += 1
                nodeAttributedDict = {
                    "node_id": output + "_buff",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": numInvertedPredecessors,
                    "fanouts": 0,
                    "depth": 0,
                    "is_sequential": False
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output + "_buff"] = idxCounter
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        elif line.__contains__("DFF"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            data_input = re.search("DFF\((.*?)\)", str(line)).group(1)
            processDFFAssignments(data_input, output, idxCounter, poList, nodeNameIDMapping, singleInputgateIOMapping, AIG_DAG)
            if output in poList:
                idxCounter += 1
            idxCounter += 1
        else:
            print(" Line contains unknown characters.", line)
            exit(1)

    for node in AIG_DAG.nodes:
        AIG_DAG.nodes[node]["fanouts"] = AIG_DAG.out_degree(node)

    compute_node_depths(AIG_DAG)

    return AIG_DAG

def dumpGMLGraph(nxCktDAG):
    gmlfileName = os.path.basename(INPUT_BENCH) + ".graphml"
    nx.write_graphml(nxCktDAG, os.path.join(GML_DUMP_LOC, gmlfileName))

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='AIGBENCH2GML', description="AIG bench to GML converter")
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('--bench', required=True, help="Path of AIG bench File")
    parser.add_argument('--gml', required=True, help="GML file dump location")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    nxCktDAG = parseAIGBenchAndCreateNetworkXGraph()
    dumpGMLGraph(nxCktDAG)

if __name__ == '__main__':
    main()