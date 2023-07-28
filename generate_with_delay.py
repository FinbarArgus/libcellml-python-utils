# %% [markdown]
# First we are going to parse the annotation file and see what we can learn...

# %%
from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS

# use this URI to identify delayed variables - not the perfect URI, but will do for now
delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

# a "readout" variable that we maybe want to connect to something external?
timecourse_readout_uri = URIRef('http://identifiers.org/mamo/MAMO_0000031')

model_name = 'delay_test'
annotation_file = f'cellml/{model_name}-updated-ids--annotations.ttl'
g = Graph().parse(annotation_file)

# find all delayed variables
variables_to_delay_info = []
for vtd in g.subjects(DCTERMS.type, variable_to_delay_uri):
    # we only expect one delay variable for each variable to delay
    dv = g.value(vtd, delay_variable_uri)
    d_amount = g.value(vtd, delay_amount_uri)
    variables_to_delay_info.append([str(vtd), str(dv), str(d_amount)])
    
print(variables_to_delay_info)

# find all timecourse readouts
readout_variables = []
for d in g.subjects(DCTERMS.type, timecourse_readout_uri):
    readout_variables.append(str(d))
    
print(readout_variables)

# %% [markdown]
# We're going to use the "model" file from the first variable to delay and only continue if all annotations use the same model...

# %%
from urllib.parse import urlparse

model_uri = variables_to_delay_info[0][0]
model_url = urlparse(model_uri)
model_file = model_url.path

delayed_ids = []
for v, dv, d_amount in variables_to_delay_info:
    url = urlparse(v)
    if url.path != model_file:
        print("found an unexpected model file for variable to delay?!")
        exit
    dv_url = urlparse(dv)
    if dv_url.path != model_file:
        print("found an unexpected model file for delay variable?!")
        exit
    d_amount_url = urlparse(d_amount)
    if d_amount_url.path != model_file:
        print("found an unexpected model file for delay variable?!")
        exit
    delayed_ids.append([url.fragment, dv_url.fragment, d_amount_url.fragment])
    
readout_ids = []
for v in readout_variables:
    url = urlparse(v)
    if url.path != model_file:
        print("found an unexpected model file for readout variable?!")
        exit
    readout_ids.append(url.fragment)

# %% [markdown]
# Now we have the model file and the IDs for the variables in that model that we want to do stuff with. So we can parse the model and see if we can find the variables.

# %%
import cellml

# on windows getting a leading '/' in the filename which libCellML doesn't like...
fixed_model_file = model_file[0:]

# parse the model in non-strict mode to allow non CellML 2.0 models
model = cellml.parse_model(fixed_model_file, False)

# and make an annotator for this model
from libcellml import Annotator
annotator = Annotator()
annotator.setModel(model)

# map our IDs to the actual variables
annotated_variables = []
for i, dv_i, d_amount_i in delayed_ids:
    # get the variable (will fail if id doesn't correspond to a variable in the model)
    v = annotator.variable(i)
    if v == None:
        print('Unable to find a variable to delay with the id {} in the given model...'.format(i))
        exit
    dv = annotator.variable(dv_i)
    if dv == None:
        print('Unable to find a delay variable with the id {} in the given model...'.format(dv_i))
        exit
    d_amount = annotator.variable(d_amount_i)
    if d_amount == None:
        print('Unable to find a delay variable with the id {} in the given model...'.format(dv_i))
        exit
    annotated_variables.append([[v, dv, d_amount], delay_variable_uri])
    
for i in readout_ids:
    # get the variable (will fail if id doesn't correspond to a variable in the model)
    v = annotator.variable(i)
    if v == None:
        print('Unable to find a readout variable with the id {} in the given model...'.format(i))
        exit
    annotated_variables.append([v, timecourse_readout_uri])

# %% [markdown]
# # TODO:
# Need to work out how to map the annotations through to the variables in the generated code....

# %% [markdown]
# Generate C code for the model.

# %%
import os 
model_dir = os.path.dirname(fixed_model_file)

# resolve imports, in non-strict mode
importer = cellml.resolve_imports(model, model_dir, False)
# need a flattened model for analysing
flat_model = cellml.flatten_model(model, importer)

from libcellml import Analyser, AnalyserModel, AnalyserExternalVariable, Generator, GeneratorProfile        

# analyse the model
a = Analyser()

# set the delayed variables as external
external_variable_info = []
for vv, uri in annotated_variables:
    if uri == delay_variable_uri:
        v = vv[0]
        dv = vv[1]
        d_amount = vv[2]
        flat_variable_to_delay = flat_model.component(v.parent().name()).variable(v.name())
        flat_delay_variable = flat_model.component(dv.parent().name()).variable(dv.name())
        flat_delay_amount_variable = flat_model.component(dv.parent().name()).variable(d_amount.name())
        aev = AnalyserExternalVariable(flat_delay_variable)
        aev.addDependency(flat_variable_to_delay)
        aev.addDependency(flat_delay_amount_variable)
        #
        # TODO: really need to work out how to handle other dependencies here to make sure 
        #       all required variables are up to date...
        #
        a.addExternalVariable(aev)
        # keep track of external variable information for use in generating code
        external_variable_info.append({
            'variable_to_delay': flat_variable_to_delay,
            'delay_variable': flat_delay_variable,
            'delay_amount_variable': flat_delay_amount_variable,
            'analyser_variable': aev
        })

a.analyseModel(flat_model)
analysed_model = a.model()
print(analysed_model.type())

# get the information for the variables to delay
for ext_variable in external_variable_info:
    ev = ext_variable['variable_to_delay']
    avs = analysed_model.variables()

    for av in avs:
        v = av.variable()
        if analysed_model.areEquivalentVariables(v, ext_variable['variable_to_delay']):
            ext_variable['variable_to_delay_index'] = av.index()
            ext_variable['state_or_variable'] = 'variable'
        if analysed_model.areEquivalentVariables(v, ext_variable['delay_variable']):
            ext_variable['delay_variable_index'] = av.index()
        if analysed_model.areEquivalentVariables(v, ext_variable['delay_amount_variable']):
            ext_variable['delay_amount_index'] = av.index()
    
    astates = analysed_model.states()
    for astate in astates:
        state = astate.variable()
        if state.name() == ext_variable['variable_to_delay'].name(): 
            ext_variable['state_to_delay_index'] = astate.index()
            ext_variable['state_or_variable'] = 'state'

# %%

# generate code from the analysed model
g = Generator()
# using the C profile to generate C code
profile = GeneratorProfile(GeneratorProfile.Profile.C)
profile.setInterfaceFileNameString(f'{model_name}.h')
g.setProfile(profile)
g.setModel(analysed_model)

preHeaderStuff = f"""
#include <stdlib.h>
#include <memory>
#include <map>
#include <sstream>

"""

preSourceStuff = f"""
#include <stddef.h>
#include <stdio.h>

EVSingleton* EVSingleton::instancePtr = NULL;
EVSingleton* evi = EVSingleton::getInstance();
"""

circularBufferHeader = f"""

class circular_buffer {{
  private:
    int size;
    int head;
    int tail;
    double *buffer;

  public:
    circular_buffer(int size);
    void put(double value);
    double get();
}};
"""

circularBuffer = f"""
// circular buffer implementation
circular_buffer::circular_buffer(int size)
{{
  this->size = size;
  this->head = 0;
  this->tail = 0;
  this->buffer = new double[size];
}}

void circular_buffer::put(double value)
{{
  buffer[head] = value;
  head = (head + 1) % size;
  if (head == tail)
    tail = (tail + 1) % size;
}}

double circular_buffer::get()
{{
  double value = buffer[tail];
  tail = (tail + 1) % size;
  return value;
}}

"""

storeEVSingletonHeader = f"""
class EVSingleton  {{
  private:
    static EVSingleton* instancePtr;
    EVSingleton() {{}}
    bool buffersInitialised = false;
  public:
    // deleting copy constructor
    EVSingleton(const EVSingleton& obj) = delete;
    bool isInitialised();

    static EVSingleton* getInstance();
    void initBuffers(double dt, double* variables);
    
    std::map<std::string, circular_buffer*> circular_buffer_dict;
    void storeVariable(int index, double value);
    double getVariable(int index);
}};
"""

# generate a global singleton class to store external variables
storeEVSingleton = f"""

void EVSingleton::initBuffers(double dt, double* variables)
{{
// Here I need to initialise circular buffers for each external variable   
"""

for ext_variable in external_variable_info:
    print(ext_variable.keys())
    index = ext_variable['delay_variable_index']
    delay_amount_index = ext_variable['delay_amount_index']
    storeEVSingleton += f'  double buffer_size = static_cast<double>(variables[{delay_amount_index}])/dt;\n' 
    storeEVSingleton += f'  circular_buffer* var_circular_buffer;\n'
    storeEVSingleton += f'  var_circular_buffer = new circular_buffer(buffer_size);\n'
    storeEVSingleton += f'  std::ostringstream ss;\n'
    storeEVSingleton += f'  ss << {index};\n'
    storeEVSingleton += f'  circular_buffer_dict[ss.str()] = var_circular_buffer;\n'
    storeEVSingleton += f'  buffersInitialised = true;\n'
    storeEVSingleton += f'  }}\n'

storeEVSingleton += f""" 

EVSingleton* EVSingleton::getInstance()
{{
  if (instancePtr == NULL)
  {{
    instancePtr = new EVSingleton();
    return instancePtr;
  }}
  else
  {{
    return instancePtr;      
  }}
}}

void EVSingleton::storeVariable(int index, double value)
{{
  // Here I need to store the value in the correct circular buffer    
  std::ostringstream ss;
  ss << index;
  circular_buffer_dict[ss.str()]->put(value);
}}

double EVSingleton::getVariable(int index)
{{
  // Here I need to get the value from the correct circular buffer
  std::ostringstream ss;
  ss << index;
  return circular_buffer_dict[ss.str()]->get();
}}

bool EVSingleton::isInitialised()
{{
  return buffersInitialised;
}}

"""

# and generate a function to compute external variables
computeEV = f"""
double externalVariable(double voi, double *states, double *variables, size_t index)
{{
  
"""
for ext_variable in external_variable_info:
    print(ext_variable.keys())

    # variable or state index
    if ext_variable["state_or_variable"] == "state":
        state_to_delay_index = ext_variable['state_to_delay_index']
    else:
        variable_to_delay_index = ext_variable['variable_to_delay_index']
        
    delay_variable_index = ext_variable['delay_variable_index']
    delay_amount_index = ext_variable['delay_amount_index']
    computeEV += f'  if (index == {delay_variable_index}) {{\n'
    computeEV += f'    if (voi < variables[{delay_amount_index}]) {{\n'
    computeEV += f'      if (evi->isInitialised() != true) {{return 0.0;}};\n'
    if ext_variable["state_or_variable"] == "state":
        computeEV += f'      evi->storeVariable({delay_variable_index}, states[{state_to_delay_index}]);\n'
    else:
        computeEV += f'      evi->storeVariable({delay_variable_index}, variables[{variable_to_delay_index}]);\n'
    computeEV += f'      return 0.0;\n'
    computeEV += f'    }} else {{;\n'
    computeEV += f'    // save the current value of the variable to the circle buffer\n'
    if ext_variable["state_or_variable"] == "state":
        computeEV += f'      evi->storeVariable({delay_variable_index}, states[{state_to_delay_index}]);\n'
    else:
        computeEV += f'      evi->storeVariable({delay_variable_index}, variables[{variable_to_delay_index}]);\n'
    computeEV += f'      double value = evi->getVariable({delay_variable_index});\n'
    computeEV += f'      return value;\n'
    computeEV += f'    }};\n'
    computeEV += f'  }}\n'

    # TODO also create an if statement for coupling variables with external models
    # if the variable is of coupling type, not a delay variable.

computeEV += f"""
  return 0.0;
}}
"""


# %%

mainScript = """
int main(void){
    double voi = 0.0;
    double end_time = 10.0;
    double dt = 0.01;
    double eps = 1e-12;
    double * states = createStatesArray();
    double * rates = createStatesArray(); // same size as states, should really be different function
    double * variables = createVariablesArray();


    // initialise buffers in the EVSingleton class

    initialiseVariables(voi, states, variables, externalVariable);
    evi->initBuffers(dt, variables); 

    computeComputedConstants(variables);


    while (voi < end_time-eps) {
        computeVariables(voi, states, rates, variables, externalVariable);
        computeRates(voi, states, rates, variables, externalVariable);

        // simple forward Euler integration
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states[i] + dt * rates[i];
        }
        
        voi += dt;   
    }
    
    
    computeVariables(voi, states, rates, variables, externalVariable);
    computeRates(voi, states, rates, variables, externalVariable);

    // TODO autogenerate a dict of variable names to print with variables

    printf("Final values:");
    printf("  time: ");
    printf("%f", voi);
    printf("  states:");
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        printf("%f\\n", states[i]);
    }
    printf("  variables:");
    for (size_t i = 0; i < VARIABLE_COUNT; ++i) {
        printf("%f\\n", variables[i]);
    }

return 0;
}
"""

with open(f'C_models/{model_name}.h', 'w') as f:

    f.write(preHeaderStuff) 
    f.write(g.interfaceCode())
    f.write(circularBufferHeader)
    f.write(storeEVSingletonHeader)


# and save to a file
with open(f'C_models/{model_name}.c', 'w') as f:

    f.write(g.implementationCode())
    f.write(preSourceStuff) # this has to be below the #include "model.h" in implementationCode()
    f.write(circularBuffer)
    f.write(storeEVSingleton)
    f.write(computeEV)
    f.write(mainScript)



