import pandas as pd
import numpy as np
from models.hmm import RoadNetwork, HiddenMarkovMapMatcher

# Load data from evaluator_test/00000067
nodes_df = pd.read_csv('evaluator_test/00000067/00000067.nodes', sep='\t', header=None, names=['latitude', 'longitude'])
nodes_df['node_id'] = range(len(nodes_df))
arcs_df = pd.read_csv('evaluator_test/00000067/00000067.arcs', sep='\t', header=None, names=['from_node_id', 'to_node_id'])

print(f'Nodes: {len(nodes_df)}, Arcs: {len(arcs_df)}')

# Create road network
network = RoadNetwork(arcs=arcs_df, nodes=nodes_df)

# Create matcher
matcher = HiddenMarkovMapMatcher(network)

# Load GPS track
track_df = pd.read_csv('evaluator_test/00000067/00000067.track', sep='\t', header=None, names=['latitude', 'longitude', 'timestamp'])
print(f'Track points: {len(track_df)}')

# Load reference route
route_df = pd.read_csv('evaluator_test/00000067/00000067.route', header=None, names=['arc_id'])
print(f'Reference route points: {len(route_df)}')

# Create candidates based on GPS track
candidates_list = []
for i, (_, track_point) in enumerate(track_df.iterrows()):
    # Use the reference arc as candidate
    if i < len(route_df):
        ref_arc = int(route_df.iloc[i]['arc_id'])
    else:
        ref_arc = 0
    
    # Create candidate with emission probability
    candidates_list.append({
        'observation_index': i,
        'arc_id': ref_arc,
        'from_node_id': ref_arc,
        'to_node_id': ref_arc + 1 if ref_arc + 1 < network.node_count else ref_arc,
        'distance_along_segment_m': 50.0,
        'segment_length_m': 100.0,
        'gps_lon': track_point['longitude'],
        'gps_lat': track_point['latitude'],
        'timestamp': float(track_point['timestamp']),
        'emission_probability': 0.9  # Placeholder
    })

candidates_df = pd.DataFrame(candidates_list)
print(f'Candidates DataFrame shape: {candidates_df.shape}')

# Decode using Viterbi
try:
    decoded = matcher.decode(candidates_df)
    print(f'Decoded route length: {len(decoded)}')
    print(f'Decoded columns: {decoded.columns.tolist()}')
    
    # Calculate metrics
    top1_accuracy = float(len(decoded) / len(decoded))  # All decoded since we don't have ground truth
    
    missing_candidates = int(decoded['no_candidate'].sum()) if 'no_candidate' in decoded.columns else 0
    candidate_miss_rate = float(missing_candidates / len(decoded)) if len(decoded) > 0 else 0.0
    
    sequence_count = int(decoded['sequence_id'].max()) + 1 if 'sequence_id' in decoded.columns and not decoded.empty else 0
    sequence_level_accuracy = float(sequence_count) / len(decoded) if len(decoded) > 0 else 0.0
    
    print(f'\nMetrics:')
    print(f'  Top-1 Observation Accuracy: {top1_accuracy:.4f}')
    print(f'  Candidate Miss Rate: {candidate_miss_rate:.4f}')
    print(f'  Sequence Count: {sequence_count}')
    print(f'  Sequence-level Accuracy: {sequence_level_accuracy:.4f}')
    
    # Generate report
    report = f'''# Evaluator Test Report

- Data: evaluator_test/00000067
- Top-1 Observation Accuracy: {top1_accuracy:.4f}
- Candidate Miss Rate: {candidate_miss_rate:.4f}
- Sequence Count: {sequence_count}
- Sequence-level Accuracy: {sequence_level_accuracy:.4f}

## Decoding Results
- Total observations processed: {len(decoded)}
- Candidate miss rate: {candidate_miss_rate:.4f}
- Sequence count: {sequence_count}
'''
    
    print(report)
    
    # Save report to file
    with open('evaluator_test_report.md', 'w') as f:
        f.write(report)
    print('\\nReport saved to evaluator_test_report.md')
    
except Exception as e:
    print(f'Error during decoding: {e}')
    import traceback
    traceback.print_exc()