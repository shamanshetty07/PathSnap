from models.hmm import RoadNetwork, HiddenMarkovMapMatcher

# arcs_df: road network arcs
# nodes_df: road network nodes
# candidates_df: candidate rows with RF probabilities

network = RoadNetwork(
    arcs=arcs_df,
    nodes=nodes_df,
)

matcher = HiddenMarkovMapMatcher(network)

decoded_route = matcher.decode(candidates_df)

print(decoded_route[
    ["observation_index", "arc_id", "emission_probability"]
])