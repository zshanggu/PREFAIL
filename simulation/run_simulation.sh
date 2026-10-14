#!/bin/bash

# Run from this directory so the relative paths below resolve
cd "$(dirname "$0")"

# Directory containing the obj files
OBJ_TYPE="sem"
OBJ_DIR="model/ur5e/assets/objects/${OBJ_TYPE}"

# Path to the XML file
XML_FILE="model/ur5e/scene.xml"

# Path to the config file
CONFIG_FILE="configs/simulation.json"

# Path to the object property file
OBJ_PROP_FILE="configs/objects/${OBJ_TYPE}.json"

# All data is saved under simulation/data/

# NPY directory containing processed ${OBJ_TYPE}; objects found here are skipped
NPY_DIR="data/processed/${OBJ_TYPE}"

# Raw data directory to clean up; must match save_path/dataset_type in CONFIG_FILE
RAW_DATA_DIR="data/train_raw/raw_data/${OBJ_TYPE}"

# File to store processed object names
PROCESSED_OBJ_FILE="data/train_raw/processed_${OBJ_TYPE}.txt"

# Backup directory for removed files (optional)
BACKUP_DIR="data/train_raw/backup"

# Number of times to repeat simulation for each object (default: 1)
NUM_REPEATS=1


# Check if directory exists
if [ ! -d "$OBJ_DIR" ]; then
  echo "Error: Directory $OBJ_DIR does not exist"
  exit 1
fi

if [ ! -d "$RAW_DATA_DIR" ]; then
  echo "Error: Directory $RAW_DATA_DIR does not exist"
  mkdir -p "$RAW_DATA_DIR"
  # exit 1
fi

# Check if XML file exists
if [ ! -f "$XML_FILE" ]; then
  echo "Error: XML file $XML_FILE does not exist"
  exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE does not exist"
  exit 1
fi

# Make sure the NPY directories exist
mkdir -p "$NPY_DIR"
mkdir -p "$BACKUP_DIR"

# Run the script to extract processed object names
# echo "Extracting processed object names from NPY files..."
python3.10 tools/list_processed_objects.py --npy_dir "$NPY_DIR" --output "$PROCESSED_OBJ_FILE"

# Ask user if they want to clean up unprocessed files
echo ""
echo "Do you want to clean up unprocessed files in $RAW_DATA_DIR? (y/n)"
read -r clean_choice

if [[ "$clean_choice" == "y" || "$clean_choice" == "Y" ]]; then
  # Ask if this should be a dry run
  echo "Do you want to do a dry run first (no actual deletion)? (y/n)"
  read -r dry_run_choice
  
  if [[ "$dry_run_choice" == "y" || "$dry_run_choice" == "Y" ]]; then
    # Do a dry run first
    echo "Performing dry run cleanup..."
    python3.10 tools/cleanup_unprocessed.py --raw_data_dir "$RAW_DATA_DIR" \
      --processed_objects_file "$PROCESSED_OBJ_FILE" \
      # --backup_dir "$BACKUP_DIR"
    
    # Ask if they want to proceed with actual deletion
    echo ""
    echo "Do you want to proceed with actual cleanup? (y/n)"
    read -r proceed_choice
    
    if [[ "$proceed_choice" == "y" || "$proceed_choice" == "Y" ]]; then
      echo "Performing actual cleanup..."
      python3.10 tools/cleanup_unprocessed.py --raw_data_dir "$RAW_DATA_DIR" \
        --processed_objects_file "$PROCESSED_OBJ_FILE" \
        --no-dry-run
        # --backup_dir "$BACKUP_DIR" \
    else
      echo "Skipping actual cleanup."
    fi
  else
    # Do the actual cleanup directly
    echo "Performing cleanup..."
    python3.10 tools/cleanup_unprocessed.py --raw_data_dir "$RAW_DATA_DIR" \
      --processed_objects_file "$PROCESSED_OBJ_FILE" \
      --backup_dir "$BACKUP_DIR" \
      --no-dry-run
  fi
fi

# Ask if user wants to continue with the simulation
echo ""
echo "Do you want to continue with the simulation? (y/n)"
read -r sim_choice

if [[ "$sim_choice" != "y" && "$sim_choice" != "Y" ]]; then
  echo "Exiting without running the simulation."
  exit 0
fi

# Ask how many times to repeat the simulation for each object
echo ""
echo "How many times do you want to repeat the simulation for each object? (default: 1)"
read -r repeat_input

if [[ -n "$repeat_input" && "$repeat_input" =~ ^[0-9]+$ ]]; then
  NUM_REPEATS=$repeat_input
fi

echo "Each object will be simulated $NUM_REPEATS time(s)"

# Count total OBJ to process
TOTAL_OBJ=$(find "$OBJ_DIR" -name "*.obj" -type f | wc -l)
echo "Total ${OBJ_TYPE} to process: $TOTAL_OBJ"
echo "Total simulations to run: $((TOTAL_OBJ * NUM_REPEATS))"

# Initialize counters
PROCESSED=0
SKIPPED=0
FAILED=0
TOTAL_SIMS=0
SUCCESSFUL_SIMS=0
FAILED_SIMS=0

# List all .obj files and process them one by one
echo "OBJ files found in $OBJ_DIR:"
find "$OBJ_DIR" -name "*.obj" -type f | while read file; do
  basename=$(basename "$file" .obj)
  echo "Processing: $basename"

  # Check if object has already been processed
  if grep -q "^$basename$" "$PROCESSED_OBJ_FILE" 2>/dev/null; then
    echo "Object $basename has already been processed. Skipping..."
    echo "-----------------------------------"
    ((SKIPPED++))
    continue
  fi

  # Call the Python script to update XML file
  python3.10 tools/update_scene_xml.py "$XML_FILE" --obj_dir "$OBJ_DIR" --obj_type "$OBJ_TYPE" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE"
  
  if [ $? -ne 0 ]; then
    echo "Error: Failed to update XML for $basename"
    echo "-----------------------------------"
    ((FAILED++))
    continue
  fi

  # Call the Python script to update config file
  python3.10 tools/update_sim_config.py "$CONFIG_FILE" --obj_type "$OBJ_TYPE" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE" --xml_file "${XML_FILE%.*}_${OBJ_TYPE}_tmp.${XML_FILE##*.}"
  
  if [ $? -ne 0 ]; then
    echo "Error: Failed to update config for $basename"
    echo "-----------------------------------"
    ((FAILED++))
    continue
  fi

  # Run the simulation NUM_REPEATS times
  for ((i=1; i<=NUM_REPEATS; i++)); do
    echo "Running simulation $i/$NUM_REPEATS for $basename..."
    ((TOTAL_SIMS++))
    
    python3.10 src/simulate_backtracking.py --config "${CONFIG_FILE%.*}_${OBJ_TYPE}_tmp.${CONFIG_FILE##*.}" --obj_type "$OBJ_TYPE"
    
    if [ $? -eq 0 ]; then
      echo "✓ Simulation $i/$NUM_REPEATS completed successfully for $basename"
      ((SUCCESSFUL_SIMS++))
    else
      echo "✗ Simulation $i/$NUM_REPEATS failed for $basename"
      ((FAILED_SIMS++))
    fi
  done
  
  ((PROCESSED++))
  echo "-----------------------------------"
done

echo ""
echo "======================================"
echo "All ${OBJ_TYPE} have been processed."
echo "======================================"
echo "${OBJ_TYPE} processed: $PROCESSED"
echo "${OBJ_TYPE} skipped: $SKIPPED"
echo "${OBJ_TYPE} failed: $FAILED"
echo "Total simulations run: $TOTAL_SIMS"
echo "Successful simulations: $SUCCESSFUL_SIMS"
echo "Failed simulations: $FAILED_SIMS"
echo "======================================"

# Make sure all files have proper permissions
chmod -R 755 model/ 2>/dev/null
chmod 644 "$CONFIG_FILE" 2>/dev/null
echo "File permissions updated."