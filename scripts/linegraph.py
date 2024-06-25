import json
import matplotlib.pyplot as plt

# Load the data
data = json.load(open('execution_times.json'))

# Initialize data structures for combined plots
lbnl_data = {algo: {"evals": [], "grouped": [], "regular": []} for algo in data.keys()}
leakdb_data = {algo: {"evals": [], "grouped": [], "regular": []} for algo in data.keys()}

# Populate the data structures
for algo in data.keys():
    for eval in sorted(data[algo]["lbnl_fdd"].keys(), key=int):
        lbnl_data[algo]["evals"].append(eval)
        lbnl_data[algo]["grouped"].append(data[algo]["lbnl_fdd"][eval]["grouped"])
        lbnl_data[algo]["regular"].append(data[algo]["lbnl_fdd"][eval]["regular"])

    for eval in sorted(data[algo]["leakdb"].keys(), key=int):
        leakdb_data[algo]["evals"].append(eval)
        leakdb_data[algo]["grouped"].append(data[algo]["leakdb"][eval]["grouped"])
        leakdb_data[algo]["regular"].append(data[algo]["leakdb"][eval]["regular"])

# Plot for lbnl_fdd
plt.figure(figsize=(12, 8))
for algo in lbnl_data.keys():
    plt.plot(lbnl_data[algo]["evals"], lbnl_data[algo]["grouped"], label=f'{algo} Grouped')
    plt.plot(lbnl_data[algo]["evals"], lbnl_data[algo]["regular"], label=f'{algo} Regular', linestyle='dashed')

plt.xlabel('Evaluations')
plt.ylabel('Execution Time (s)')
plt.title('Execution Times for lbnl_fdd Dataset')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('lbnl_fdd_execution_times.png')
plt.show()
# plt.savefig('/mnt/data/lbnl_fdd_combined_execution_times.png')

# Plot for leakdb
plt.figure(figsize=(12, 8))
for algo in leakdb_data.keys():
    plt.plot(leakdb_data[algo]["evals"], leakdb_data[algo]["grouped"], label=f'{algo} Grouped')
    plt.plot(leakdb_data[algo]["evals"], leakdb_data[algo]["regular"], label=f'{algo} Regular', linestyle='dashed')

plt.xlabel('Evaluations')
plt.ylabel('Execution Time (s)')
plt.title('Execution Times for leakdb Dataset')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('leakdb_execution_times.png')
plt.show()
# plt.savefig('/mnt/data/leakdb_combined_execution_times.png')

# Display the paths to the saved plots
combined_output_files = ['lbnl_fdd_combined_execution_times.png', 'leakdb_combined_execution_times.png']
combined_output_files
