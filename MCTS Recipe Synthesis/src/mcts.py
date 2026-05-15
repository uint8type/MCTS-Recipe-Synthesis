import torch
import math
import random
import copy
import numpy as np
from torch_geometric.data import Batch, Data

# Import your existing model structures
from model import SynthNet, NodeEncoder, SynthFlowEncoder

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    State: A partial list of synthesis actions (recipe).
    """
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # The action taken to reach this node (int 0-6)
        self.children = {}    # Map: action_index -> MCTSNode
        self.visits = 0       # N
        self.value_sum = 0.0  # Q (Accumulated reward)
        
    def is_fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)

    def is_leaf(self):
        return len(self.children) == 0

    def get_ucb_score(self, c_param=1.414):
        if self.visits == 0:
            return float('inf')
        
        # Q_value is average reward. 
        # Since we use negative power as reward, higher is better (lower power).
        q_value = self.value_sum / self.visits
        u_value = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return q_value + u_value

    def best_child(self, c_param=1.414):
        # Select child with highest UCB score
        choices = [
            (child.get_ucb_score(c_param), action, child)
            for action, child in self.children.items()
        ]
        # Maximize UCB
        return max(choices, key=lambda x: x[0])[2]


class SynthesisMCTS:
    def __init__(self, model, initial_graph_data, device, 
                 recipe_length=20, num_actions=7, exploration_c=1.414):
        """
        Args:
            model: The trained SynthNet model (in eval mode).
            initial_graph_data: The PyG Data object for the specific AIG design.
            device: 'cuda' or 'cpu'.
            recipe_length: Fixed length of recipe model expects (20).
            num_actions: Number of transformation types (0-6).
        """
        self.model = model
        self.graph_data = initial_graph_data
        self.device = device
        self.target_len = recipe_length
        self.actions = list(range(num_actions))
        self.c_param = exploration_c
        self.min_power = float('inf')
        self.max_power = float('-inf') 

    def search(self, iterations=5000):
        """
        Runs MCTS for a specified number of iterations.
        Returns: The best recipe (list of ints) found.
        """
        root = MCTSNode()
        
        for _ in range(iterations):
            node = root
            current_recipe = []

            # 1. SELECTION
            # Traverse down until we find a node that isn't fully expanded or is a terminal state
            while not node.is_leaf() and node.is_fully_expanded(self.actions):
                node = node.best_child(self.c_param)
                current_recipe.append(node.action)
                
            # Stop if path exceeds target length (should not happen if logic is correct)
            if len(current_recipe) >= self.target_len:
                reward = self.get_prediction_reward(current_recipe)
                self.backpropagate(node, reward)
                continue

            # 2. EXPANSION
            # If not fully expanded, add a new child
            if not node.is_fully_expanded(self.actions):
                unexplored_actions = [a for a in self.actions if a not in node.children]
                action = random.choice(unexplored_actions)
                new_node = MCTSNode(parent=node, action=action)
                node.children[action] = new_node
                node = new_node
                current_recipe.append(action)

            # 3. SIMULATION (ROLLOUT)
            # Randomly fill the rest of the recipe to length 20
            remaining_steps = self.target_len - len(current_recipe)
            rollout_recipe = current_recipe + random.choices(self.actions, k=remaining_steps)
            
            # Predict Power using the Neural Net
            reward = self.get_prediction_reward(rollout_recipe)

            # 4. BACKPROPAGATION
            self.backpropagate(node, reward)

        # Return best sequence from root based on most visits (robustness)
        # Or best average value (exploitation)
        return self.get_best_sequence(root)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    def get_prediction_reward(self, recipe):
        """
        Wraps the model prediction.
        Returns NEGATIVE power (because MCTS maximizes reward, but we want min power).
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare batch
            # We must clone the graph data to avoid mutating the original
            data_clone = self.graph_data.clone()
            
            # Update synVec with the rollout recipe
            # Model expects shape [batch_size, sequence_length]
            recipe_tensor = torch.tensor(recipe, dtype=torch.long).unsqueeze(0) # [1, 20]
            
            # The original model code expects synVec in the data object
            # Note: In model.py, forward does: synthFlow = batch_data.synVec
            # And: h_syn = self.synth_encoder(synthFlow.reshape(-1,20))
            data_clone.synVec = recipe_tensor
            
            # Create a Batch object (batch size 1)
            batch = Batch.from_data_list([data_clone]).to(self.device)
            
            # # Predict
            # pred_power = self.model(batch)
            
            # # Extract scalar value
            # power_val = pred_power.item()
            
            # # Reward: -Power (Maximize negative power = Minimize positive power)
            # # You might want to scale this if power values are very large/small
            # return -power_val
            pred_power = self.model(batch).item()

            # Update global bounds observed so far
            self.min_power = min(self.min_power, pred_power)
            self.max_power = max(self.max_power, pred_power)

            # Standardize Reward: 
            # We want LOW power to be HIGH reward (1.0)
            # and HIGH power to be LOW reward (0.0)
            if self.max_power == self.min_power:
                return 0.5 
            
            normalized_reward = (self.max_power - pred_power) / (self.max_power - self.min_power)
            return normalized_reward

    def get_best_sequence(self, root):
        """
        Traverses the tree greedily based on visit counts to reconstruct the best path.
        """
        node = root
        best_recipe = []
        
        while len(best_recipe) < self.target_len:
            if not node.children:
                # If tree doesn't go deep enough, fill with random or break
                break
                
            # Select child with most visits (standard MCTS final selection)
            best_action = max(node.children.items(), key=lambda item: item[1].visits)[0]
            best_recipe.append(best_action)
            node = node.children[best_action]
            
        # If the search didn't fully explore to depth 20, fill the rest randomly
        # if len(best_recipe) < self.target_len: Akash comment these 3 lines
        #     remaining = self.target_len - len(best_recipe)
        #     best_recipe.extend(random.choices(self.actions, k=remaining))
            
        return best_recipe



def run_mcts_optimization(model_path, data_path, device_name='cuda'):
    # 1. Setup Device
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    # 2. Load Data (Simplified for example)
    # In reality, you load this using your NetlistGraphDataset
    # Here we mock a single data object based on your schema
    print("Loading Graph Data...")
    
    # Placeholder: You must load your actual Data object here
    # Example: dataset = NetlistGraphDataset(...)
    # graph_data = dataset[0]
    
    # Creating a dummy graph for demonstration if you run this script directly
    graph_data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        node_type=torch.tensor([0, 1], dtype=torch.long),
        num_inverted_predecessors=torch.tensor([0, 0], dtype=torch.long),
        fanouts=torch.tensor([1, 1], dtype=torch.float),
        depth=torch.tensor([0, 1], dtype=torch.float),
        synVec=torch.zeros((20), dtype=torch.long), # Placeholder
        desName=["test_design"],
        synID=[0]
    )

    # 3. Initialize Model
    print("Initializing Model...")
    # These dimensions must match your trained model
    node_emb_dim = 3
    synth_emb_dim = 3
    synth_flow_dim = 20 * synth_emb_dim # 60
    
    node_encoder = NodeEncoder(emb_dim=node_emb_dim)
    synth_encoder = SynthFlowEncoder(emb_dim=synth_emb_dim)
    
    model = SynthNet(
        node_encoder=node_encoder,
        synth_encoder=synth_encoder,
        n_classes=1, # Regression output
        synth_input_dim=synth_flow_dim,
        node_input_dim=node_emb_dim + 2 # + fanout/depth handled internally? Check model.py GNN_node
        # Note: In model.py GNN_node forward: cat([node_type, num_inv, fanout, depth])
        # So node_input_dim usually 3 if embedding size is 3? 
        # Actually in model.py: self.gnn = GNN(..., self.node_enc_outdim+1)
    )
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    except:
        print("Warning: Could not load weights from path. Using random weights for demo.")

    model.to(device)

    # 4. Run MCTS
    print("Starting MCTS Search...")
    mcts = SynthesisMCTS(
        model=model,
        initial_graph_data=graph_data,
        device=device,
        recipe_length=20,
        num_actions=7,    # Actions 0-6
        exploration_c=1.5 # Tuning parameter for exploration
    )
    
    # Run for 500 iterations
    best_recipe = mcts.search(iterations=500)
    
    print("\nOptimization Complete.")
    print(f"Best Recipe Found: {best_recipe}")
    
    # Evaluate Best Recipe
    reward = mcts.get_prediction_reward(best_recipe)
    print(f"Predicted Power for Best Recipe: {-reward:.4f}")

if __name__ == "__main__":
    # You can point this to your actual model path
    run_mcts_optimization("gcn-epoch-best.pt", "data_path")