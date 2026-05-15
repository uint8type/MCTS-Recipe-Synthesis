#!/bin/bash
set -e   # Stop immediately if any command fails

DESIGN="cavlc"
HOME_DIR="$HOME"

echo "STEP 1: Generate synthesis scripts"
python automate_synthesisScriptGen1.py \
  --home "$HOME_DIR" \
  --script "$HOME_DIR/OpenABC_DATA/synScripts/" \
  --lib "$HOME_DIR/OpenABC_DATA/lib/sky130.lib" \
  --designs "$DESIGN"

echo "STEP 2: Bulk synthesis"
python automate_bulkSynthesis.py \
  --home "$HOME_DIR" \
  --designs "$DESIGN"

echo "STEP 3: Run synthesis shell script"
chmod +x "$HOME_DIR/OpenABC_DATA/bench/synthesisBulk_${DESIGN}.sh"
"$HOME_DIR/OpenABC_DATA/bench/synthesisBulk_${DESIGN}.sh"

echo "STEP 4: Convert bench to GraphML "
python automate_synbench2Graphml.py \
  --home "$HOME_DIR" \
  --designs "$DESIGN"

echo "STEP 5: Run GraphML shell script"
chmod +x "$HOME_DIR/OpenABC_DATA/graphml/${DESIGN}_GMLGeneration.sh"
"$HOME_DIR/OpenABC_DATA/graphml/${DESIGN}_GMLGeneration.sh"

echo "STEP 6: Generate PyG AIG data"
python PyGDataAIG.py \
  --ptdata "$HOME_DIR/OpenABC_DATA/ptdata/" \
  --designs "$DESIGN" \
  --gs "$HOME_DIR/OpenABC_DATA/graphml/${DESIGN}" \
  --synvec "$HOME_DIR/OpenABC_DATA/synthID2Vec.pickle"

echo "STEP 7: Collect area and delay"
python collectAreaAndDelay.py \
  --home "$HOME_DIR" \
  --designs "$DESIGN"

echo "STEP 8: Collect graph statistics"
python collectGraphStatistics.py \
  --gml "$HOME_DIR/OpenABC_DATA/graphml/" \
  --des "$DESIGN" \
  --stats "$HOME_DIR/OpenABC_DATA/statistics/"

echo "STEP 9: Pickle stats for ML"
python pickleStatsForML.py \
  --stats "$HOME_DIR/OpenABC_DATA/statistics/"

echo "PIPELINE COMPLETED SUCCESSFULLY"
