# This file will be moved to circulatory autogen and turned into a class for labelling cellml
# models with specification from the module_config.json file. This will be done in the parsers.

# import standard stuff...
import os
import sys
import cellml

# TODO change the below two lines so I get it from the inputs
model_dir = '../circulatory_autogen/generated_models/cpp_coupling' 
file_name = 'cpp_coupling'
file_name_with_ids = f'{file_name}-with-ids'

model_file_path = os.path.join(model_dir, file_name + '.cellml')
model_file_path_with_ids = os.path.join(model_dir, file_name_with_ids + '.cellml')
model = cellml.parse_model(model_file_path, False)

from libcellml import Annotator
annotator = Annotator()
annotator.setModel(model)

if annotator.assignAllIds():
    print('Some entities have been assigned an ID, you should save the model!')
else:
    print('Everything already had an ID.')

duplicates = annotator.duplicateIds()
if len(duplicates) > 0:
    print("There are some duplicate IDs, behaviour may be unreliable...")
    print(duplicates)

# blow away all the IDs and reassign them
annotator.clearAllIds()
annotator.assignAllIds()
model_string = cellml.print_model(model)
print(model_string)

# and save the updated model to a new file (currently with the same name) 
# - note, we need the model filename for making our annotations later
with open(model_file_path_with_ids, 'w') as f:
    f.write(model_string)

# get the ID of the variables we want to annotate
# TODO This will be obtained from the parsed vessel_array and module_config files
variables_to_delay = []
delay_variables = []
delay_amounts = []
independent_variables = []

# TODO in circulatory autogen we will need to figure out what variables are input and output by what
# modules are coupled to a Cpp module.

input_variables = [model.component('main_vessel').variable('v_out').id()]
output_variables = [model.component('main_vessel').variable('u').id()]


# make an RDF Graph to add annotations to - using rdflib
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS

g = Graph()

# define some URIs for things we need

# use this URI to identify delayed variables - not the perfect URI, but will do for now
#     This is actually "delayed differential equation model" from the MAMO ontology
#delay_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000089')
delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
independent_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

# use this for some random thing that I want to define - http://example.com is a good base for things that will never resolve
stuff_uri = URIRef('http://example.com/cool-thing#21')

# a "readout" variable that we maybe want to connect to something external?
timecourse_readout_uri = URIRef('http://identifiers.org/mamo/MAMO_0000031')
output_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000018')
input_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000017')

# Create an RDF URI node for our variable to use as the subject for multiple triples
# note: we are going to serialise the RDF graph into the same folder, so we need a URI that is relative to the intended file
variable_to_delay_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + variable) for variable in variables_to_delay]
delay_variable_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + variable) for variable in delay_variables]
delay_amount_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + delay_amount) for delay_amount in delay_amounts]
independent_variable_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + variable) for variable in independent_variables]

output_variable_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + variable) for variable in output_variables]
input_variable_name_uri = [URIRef(f'{file_name_with_ids}.cellml' + '#' + variable) for variable in input_variables]

# Add triples using store's add() method.
# We're using the Dublin Core term "type" to associate the variable with the delay...
for II in range(len(variable_to_delay_name_uri)):
    g.add((variable_to_delay_name_uri[II], DCTERMS.type, variable_to_delay_uri[II]))
    g.add((variable_to_delay_name_uri[II], delay_variable_uri[II], delay_variable_name_uri[II]))
    g.add((variable_to_delay_name_uri[II], independent_variable_uri[II], independent_variable_name_uri[II]))
    g.add((variable_to_delay_name_uri[II], delay_amount_uri[II], delay_amount_name_uri[II]))
# Set coupling variables
for II in range(len(output_variables)):
    g.add((output_variable_name_uri[II], DCTERMS.type, output_variable_uri))
for II in range(len(input_variables)):
    g.add((input_variable_name_uri[II], DCTERMS.type, input_variable_uri))

# print all the data in the turtle format
print(g.serialize(format='ttl'))

# and save to a file
with open(os.path.join(model_dir, file_name_with_ids + '--annotations.ttl'), 'w') as f:
    f.write(g.serialize(format='ttl'))
