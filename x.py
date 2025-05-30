from graphviz import Digraph

# Create a new directed graph for the flowchart
dot_flowchart = Digraph(comment='Flowchart: Bird and Drone Classification System')

# Define flowchart nodes with appropriate shapes
dot_flowchart.node('Start', 'Start', shape='oval')
dot_flowchart.node('1', 'Input Dataset (DIAT-uSAT_dataset)', shape='box')
dot_flowchart.node('2', 'Preprocess Data\n(Augmentation, Resizing)', shape='box')
dot_flowchart.node('3', 'Train CNN Model', shape='box')
dot_flowchart.node('4', 'Evaluate Model\n(Save Metrics, Visualizations)', shape='box')
dot_flowchart.node('5', 'Save Trained Model', shape='box')
dot_flowchart.node('6', 'User Inputs Image(s)\n(Single or Batch)', shape='box')
dot_flowchart.node('7', 'Load Trained Model', shape='box')
dot_flowchart.node('8', 'Preprocess Input Image(s)', shape='box')
dot_flowchart.node('9', 'Predict and Aggregate Results\n(Binary: Bird/Drone)', shape='box')
dot_flowchart.node('10', 'Visualize and Save Results', shape='box')
dot_flowchart.node('End', 'End', shape='oval')

# Define flowchart edges (sequential flow)
dot_flowchart.edge('Start', '1')
dot_flowchart.edge('1', '2')
dot_flowchart.edge('2', '3')
dot_flowchart.edge('3', '4')
dot_flowchart.edge('4', '5')
dot_flowchart.edge('5', '6')
dot_flowchart.edge('6', '7')
dot_flowchart.edge('7', '8')
dot_flowchart.edge('8', '9')
dot_flowchart.edge('9', '10')
dot_flowchart.edge('10', 'End')

# Render the flowchart
dot_flowchart.render('flowchart', view=True, format='png', cleanup=True)