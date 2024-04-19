with open('features_enhancement2.txt', 'r') as file:
    lines = file.readlines()

processed_lines = []
for line in lines:
    values = line.split()
    # Make sure to save seven values per row
    if len(values) < 7:
        values += ['0'] * (7 - len(values))
    processed_lines.append(','.join(values[:7]))

with open('features_enhancement2.txt', 'w') as file:
    file.write('\n'.join(processed_lines))
    print("successfully!")