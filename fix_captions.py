import csv

input_file = 'data/Softnet/01/captions.robot.csv'
output_file = 'data/Softnet/01/captions_simple.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow(['modelId', 'caption'])
    
    for row in reader:
        writer.writerow([row['modelId'], row['description']])

print(f"Created {output_file}")
