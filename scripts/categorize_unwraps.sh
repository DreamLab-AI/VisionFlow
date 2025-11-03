#!/bin/bash
# Categorize all unwrap() calls

OUTPUT_DIR="/home/devuser/workspace/project/docs/migration"
mkdir -p "$OUTPUT_DIR"

echo "=== Unwrap Categorization Report ===" > "$OUTPUT_DIR/categorization.txt"
echo "Date: $(date)" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# Number::from_f64().unwrap()
echo "1. Number::from_f64().unwrap() - JSON number conversions:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "Number::from_f64.*\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/json_number_unwraps.txt"
wc -l "$OUTPUT_DIR/json_number_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# .lock().unwrap()
echo "2. .lock().unwrap() - Mutex locks:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.lock()\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/mutex_unwraps.txt"
wc -l "$OUTPUT_DIR/mutex_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# .read().unwrap()
echo "3. .read().unwrap() - RwLock reads:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.read()\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/rwlock_read_unwraps.txt"
wc -l "$OUTPUT_DIR/rwlock_read_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# .write().unwrap()
echo "4. .write().unwrap() - RwLock writes:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.write()\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/rwlock_write_unwraps.txt"
wc -l "$OUTPUT_DIR/rwlock_write_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# .parse().unwrap()
echo "5. .parse().unwrap() - String parsing:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.parse()\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/parse_unwraps.txt"
wc -l "$OUTPUT_DIR/parse_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# .get().unwrap()
echo "6. .get().unwrap() - Collection access:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.get(.*\.unwrap()" src/ --include="*.rs" -n | grep -v test > "$OUTPUT_DIR/collection_unwraps.txt"
wc -l "$OUTPUT_DIR/collection_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# Other unwraps
echo "7. Other unwrap() calls:" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.unwrap()" src/ --include="*.rs" -n | grep -v test | \
  grep -v "Number::from_f64" | \
  grep -v "\.lock()\.unwrap()" | \
  grep -v "\.read()\.unwrap()" | \
  grep -v "\.write()\.unwrap()" | \
  grep -v "\.parse()\.unwrap()" > "$OUTPUT_DIR/other_unwraps.txt"
wc -l "$OUTPUT_DIR/other_unwraps.txt" >> "$OUTPUT_DIR/categorization.txt"
echo "" >> "$OUTPUT_DIR/categorization.txt"

# Total count
echo "=== TOTAL ===" >> "$OUTPUT_DIR/categorization.txt"
grep -r "\.unwrap()" src/ --include="*.rs" | grep -v test | grep -v "// SAFETY" | wc -l >> "$OUTPUT_DIR/categorization.txt"

cat "$OUTPUT_DIR/categorization.txt"
