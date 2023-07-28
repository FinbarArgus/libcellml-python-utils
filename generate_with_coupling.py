# First we are going to parse the annotation file and see what we can learn...

import os
from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS

# solver info
# solver = 'forward_euler'
# solver = 'RK4' # TODO include in circulatory autogen yaml file
solver = 'CVODE' # 
create_main = True
couple_to_1d = True
# external_headers = ['cpp_external.h']
external_headers = []
if couple_to_1d:
    external_headers += ['model1d.h']


# delay variables uri
delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
independent_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

# identifiers for input and output variables that we want to connect to another model
output_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000018')
input_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000017')

model_dir = '/home/farg967/Documents/git_projects/circulatory_autogen/generated_models/cpp_coupling' # '/home/farg967/Documents/git_projects/libcellml-python-utils/test_cases' 
model_name = 'cpp_coupling'
model_file_path = os.path.join(model_dir, model_name + '-with-ids.cellml')
annotation_file = os.path.join(model_dir, f'{model_name}-with-ids--annotations.ttl')
g = Graph().parse(annotation_file)

# find all delayed variables
variables_to_delay_info = []
for vtd in g.subjects(DCTERMS.type, variable_to_delay_uri):
    # we only expect one delay variable for each variable to delay
    dv = g.value(vtd, delay_variable_uri)
    d_amount = g.value(vtd, delay_amount_uri)
    variables_to_delay_info.append([str(vtd), str(dv), str(d_amount)])
    
print(variables_to_delay_info)

# # find all timecourse readouts
# readout_variables = []
# for d in g.subjects(DCTERMS.type, timecourse_readout_uri):
#     readout_variables.append(str(d))
    
# print(readout_variables)

# find input and output variables
input_variable_info= []
output_variable_info= []
for d in g.subjects(DCTERMS.type, input_variable_uri):
    input_variable_info.append(str(d))
for d in g.subjects(DCTERMS.type, output_variable_uri):
    output_variable_info.append(str(d))
    
print('input_variables')
print(input_variable_info)
print('output_variables')
print(output_variable_info)

# We're going to use the "model" file from the first variable to delay and only continue if all annotations use the same model...

from urllib.parse import urlparse

delayed_ids = []
for vtd, dv, d_amount in variables_to_delay_info:
    vtd_url = urlparse(vtd)
    if vtd_url.path != model_file_path:
        print("found an unexpected model file for variable to delay?!")
        exit()
    dv_url = urlparse(dv)
    if dv_url.path != model_file_path:
        print("found an unexpected model file for delay variable?!")
        exit()
    d_amount_url = urlparse(d_amount)
    if d_amount_url.path != model_file_path:
        print("found an unexpected model file for delay amount?!")
        exit()
    delayed_ids.append([url.fragment, dv_url.fragment, d_amount_url.fragment])
    
# readout_ids = []
# for v in readout_variables:
#     url = urlparse(v)
#     if url.path != model_file:
#         print("found an unexpected model file for readout variable?!")
#         exit
#     readout_ids.append(url.fragment)

input_variable_ids = []
for input_var in input_variable_info:
    input_var_url = urlparse(input_var)
    if input_var_url.path != model_file_path:
        print("found an unexpected model file for readout variable?!")
        exit()
    input_variable_ids.append(input_var_url.fragment)

output_variable_ids = []
for output_var in output_variable_info:
    output_var_url = urlparse(output_var)
    if output_var_url.path != model_file_path:
        print("found an unexpected model file for readout variable?!")
        exit()
    output_variable_ids.append(output_var_url.fragment)

# Now we have the model file and the IDs for the variables in that model that we want to do stuff with. So we can parse the model and see if we can find the variables.

import cellml

# on windows getting a leading '/' in the filename which libCellML doesn't like...
fixed_model_file_path = model_file_path[0:]

# parse the model in non-strict mode to allow non CellML 2.0 models
model = cellml.parse_model(fixed_model_file_path, False)

# and make an annotator for this model
from libcellml import Annotator
annotator = Annotator()
annotator.setModel(model)

# map our IDs to the actual variables
annotated_variables = []
for vtd_id, dv_id, d_amount_id in delayed_ids:
    # get the variable (will fail if id doesn't correspond to a variable in the model)
    vtd = annotator.variable(vtd_id)
    if vtd == None:
        print('Unable to find a variable to delay with the id {} in the given model...'.format(vtd_id))
        exit()
    dv = annotator.variable(dv_id)
    if dv == None:
        print('Unable to find a delay variable with the id {} in the given model...'.format(dv_id))
        exit()
    d_amount = annotator.variable(d_amount_id)
    if d_amount == None:
        print('Unable to find a delay variable with the id {} in the given model...'.format(dv_id))
        exit()
    annotated_variables.append([[vtd, dv, d_amount], delay_variable_uri])
    
# for i in readout_ids:
#     # get the variable (will fail if id doesn't correspond to a variable in the model)
#     v = annotator.variable(i)
#     if v == None:
#         print('Unable to find a readout variable with the id {} in the given model...'.format(i))
#         exit
#     annotated_variables.append([v, timecourse_readout_uri])

for input_var_id in input_variable_ids:
    # get the variable (will fail if id doesn't correspond to a variable in the model)
    input_var = annotator.variable(input_var_id)
    if input_var == None:
        print('Unable to find a readout variable with the id {} in the given model...'.format(input_var_id))
        exit()
    annotated_variables.append([input_var, input_variable_uri])

for output_var_id in output_variable_ids:
    # get the variable (will fail if id doesn't correspond to a variable in the model)
    output_var = annotator.variable(output_var_id)
    if output_var == None:
        print('Unable to find a readout variable with the id {} in the given model...'.format(output_var_id))
        exit()
    annotated_variables.append([output_var, output_variable_uri])

# Need to work out how to map the annotations through to the variables in the generated code....
# Generate C code for the model.

import os 
model_dir = os.path.dirname(fixed_model_file_path)

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
            'variable': flat_variable_to_delay,
            'delay_variable': flat_delay_variable,
            'delay_amount_variable': flat_delay_amount_variable,
            'analyser_variable': aev,
            'variable_type': 'delay'
        })
    elif uri == input_variable_uri:
        v = vv
        input_variable = flat_model.component(v.parent().name()).variable(v.name())
        aev = AnalyserExternalVariable(input_variable)
        a.addExternalVariable(aev)
        external_variable_info.append({
            'variable': input_variable,
            'analyser_variable': aev,
            'variable_type': 'input'
        })
        # TODO I need to include specification of the external variable here? i.e. the name 
        # or index of the variable from the Cpp code?
    elif uri == output_variable_uri:
        v = vv
        output_variable = flat_model.component(v.parent().name()).variable(v.name())
        external_variable_info.append({
            'variable': output_variable,
            'variable_type': 'output'
        })
        # TODO I need to include specification of the external variable here? i.e. the name 
        # or index of the variable from the Cpp code?

a.analyseModel(flat_model)
analysed_model = a.model()
print(analysed_model.type())
if analysed_model.type() != AnalyserModel.Type.ODE:
    print("model is not a valid ODE model, aborting...")
    exit()
# if not analysed_model.isValid():
#     print("model is not valid, aborting...")
#     exit()

# get the information for the variables to delay
for ext_variable in external_variable_info:
    ev = ext_variable['variable']
    avs = analysed_model.variables()

    for av in avs:
        v = av.variable()
        if analysed_model.areEquivalentVariables(v, ext_variable['variable']):
            ext_variable['variable_index'] = av.index()
            ext_variable['state_or_variable'] = 'variable'

        if ext_variable['variable_type'] == 'delay':
            if analysed_model.areEquivalentVariables(v, ext_variable['delay_variable']):
                ext_variable['delay_variable_index'] = av.index()
            if analysed_model.areEquivalentVariables(v, ext_variable['delay_amount_variable']):
                ext_variable['delay_amount_index'] = av.index()
    
    astates = analysed_model.states()
    for astate in astates:
        state = astate.variable()
        if state.name() == ext_variable['variable'].name(): 
            ext_variable['state_index'] = astate.index()
            ext_variable['state_or_variable'] = 'state'

# generate code from the analysed model
g = Generator()
# using the C profile to generate C code
profile = GeneratorProfile(GeneratorProfile.Profile.C)
#profile.setInterfaceFileNameString(f'{model_name}.h')
profile.setInterfaceFileNameString(f'model0d.h')
g.setProfile(profile)
g.setModel(analysed_model)

preHeaderStuff = f"""
#include <stdlib.h>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <functional>

"""
for external_header in external_headers:
    preHeaderStuff += f'#include "{external_header}"\n'

if solver == 'CVODE':
    preHeaderStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 

"""

interFaceCodePreClass = ''
interFaceCodeInClass = ''
pre_class = True
for line in g.interfaceCode().split('\n'):
    if 'extern ' in line:
        if 'VERSION' in line:
            line = line.replace('VERSION', 'VERSION_')
        if 'VERSION' not in line:
            line = line.replace('extern ', '')
    if '(* ExternalVariable)' in line:
        continue
    if ', ExternalVariable externalVariable' in line:
        line = line.replace(', ExternalVariable externalVariable', '')
    if 'const VariableInfo' in line:
        line = 'static ' + line
    if line.startswith('const size_t STATE_COUNT'):
        pre_class = False
    if pre_class:
        interFaceCodePreClass += line + '\n'
    else:
        interFaceCodeInClass += '    ' + line + '\n'
        

if solver == 'CVODE':
    classInitHeader = """
//forward declare the userOdeData class
class UserOdeData;
"""

classInitHeader += """
// this is the the 0D model class definition
// it contains the model variables and the functions to compute the rates and variables
class Model0d {
public:
    // constructor
    Model0d();
    // destructor
    ~Model0d();
"""

otherHeaderInits = f"""
    double externalVariable(double voi, double *states, double *rates, double *variables, size_t index);
    void solveOneStep(double dt);
    double voi;
    double voiEnd;
    double eps;
    double * states;
    double * rates;
    double * variables;
"""
if couple_to_1d:
    otherHeaderInits += """
    std::map<std::string, std::map<std::string, int>> cellml_index_to_vessel1d_info;
    std::vector<std::map<std::string, int>> vessel1d_info;
    Model1d * model1d_ptr;
    int num_vessel1d_connections;
    """


if solver == 'RK4':
    otherHeaderInits += """
    double * k1;
    double * k2;
    double * k3;
    double * k4;
    double * temp_states;
    """

if solver == 'CVODE':
    otherHeaderInits += """
    SUNContext context;
    void *solver;
    N_Vector y; 
    UserOdeData *userData= nullptr;
    SUNMatrix matrix;
    SUNLinearSolver linearSolver;
    using FunctionType = std::function<void(double, double*, double*, double*)>;
"""
# using computeRatesType = void (*)(double, double *, double *, double *);
# static int func(double voi, N_Vector y, N_Vector ydot, void *userData);

classFinisherHeader = """
};
"""

preSourceStuff = f"""
#include <stddef.h>
#include <stdio.h>
"""
if solver == 'CVODE':
    preSourceStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
"""



# split implementation code in two so we can change it into a class
preClassStuff = ''
classInit = """
Model0d::Model0d() :
"""
postClassInit = ''
classInit += """    voi(0.0),
    eps(1e-08),
    states(nullptr),
    rates(nullptr),
    variables(nullptr),
"""

if solver == 'RK4':
    classInit += """    k1(nullptr),
    k2(nullptr),
    k3(nullptr),
    k4(nullptr),
    temp_states(nullptr),
    """
    
if couple_to_1d:
    classInit += """    model1d_ptr(nullptr),
"""
# create mapping between external variable index and 1D vessel and 
# BC information
for ext_variable in external_variable_info:
    # Find out whether this is coupled to a 1D vessel from
    # the vessel array
    if ext_variable['variable_index'] == 5: # TODO find this
        ext_variable['coupled_to_type'] = 'vessel1d'
    else:
        ext_variable['coupled_to_type'] = None 

    if ext_variable['coupled_to_type'] == 'vessel1d':
        # find out what 1D vessel this variable is associated with
        # from the vessel_array TODO
        ext_variable['vessel1d_index'] = 0 # TODO find this
        ext_variable['flow_or_pressure_bc'] = 'flow' # TODO find this
        # find out if this variable is an inlet or outlet
        # 0 for inlet, 1 for outlet
        ext_variable['bc_inlet0_or_outlet1'] = 0 # TODO find this

classInit += """    cellml_index_to_vessel1d_info{
"""
num_vessel1d_connections = 0 
for idx, ext_variable in enumerate(external_variable_info):
    if idx != 0:
        classInit += ',\n'
    if ext_variable['coupled_to_type'] == 'vessel1d':
        classInit += f"""       {{ "{ext_variable["variable_index"]}", {{ 
            {{ "vessel1d_idx", {ext_variable["vessel1d_index"]}}}, 
            {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }} 
        }}}}"""
        num_vessel1d_connections += 1
classInit += '    },\n'
classInit += f'    num_vessel1d_connections({num_vessel1d_connections}),\n'

# now create a vessel1d_info vector of dicts for each connected variable
classInit += """    vessel1d_info{ """
for idx, ext_variable in enumerate(external_variable_info):
    if idx != 0:
        classInit += ',\n'
    if ext_variable['coupled_to_type'] == 'vessel1d':
        classInit += f"""       
        {{  {{ "cellml_idx", {ext_variable["variable_index"]}}}, 
            {{ "vessel1d_idx", {ext_variable["vessel1d_index"]}}}, 
            {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }} 
        }}"""
classInit += '    },\n'


# TODO check the below is works for delays
# or remove the singleton and just have a buffer
# within the class
if len(variables_to_delay_info) > 0:
    classInit += '    evi(EVSingleton::getInstance()),\n'

pre_class = True
in_class_init = False
for line in g.implementationCode().split('\n'):
    if line.startswith('const size_t STATE_COUNT'):
        pre_class = False
        in_class_init = True

    if 'VERSION' in line:
        line = line.replace('VERSION', 'VERSION_')
    if line.startswith('double * createStatesArray'):
        in_class_init = False

    if pre_class:
        preClassStuff += line + '\n'
    elif in_class_init:
        if 'const size_t STATE_COUNT' in line:
            line = line.replace('const size_t STATE_COUNT = ', 'STATE_COUNT(')
            line = line.replace(';', '),')
        if 'const size_t VARIABLE_COUNT' in line:
            line = line.replace('const size_t VARIABLE_COUNT = ', 'VARIABLE_COUNT(')
            line = line.replace(';', ') {')
        classInit += '    ' + line + '\n'
    else:
        if 'createStatesArray' in line:
            line = line.replace('createStatesArray', 'Model0d::createStatesArray')
        if 'createVariablesArray' in line:
            line = line.replace('createVariablesArray', 'Model0d::createVariablesArray')
        if 'deleteArray' in line:
            line = line.replace('deleteArray', 'Model0d::deleteArray')
        if 'initialiseVariables' in line:
            line = line.replace('initialiseVariables', 'Model0d::initialiseVariables')
        if 'computeComputedConstants' in line:
            line = line.replace('computeComputedConstants', 'Model0d::computeComputedConstants')
        if 'computeRates' in line:
            line = line.replace('computeRates', 'Model0d::computeRates')
        if 'computeVariables' in line:
            line = line.replace('computeVariables', 'Model0d::computeVariables')
        if 'states' in line:
            line = line.replace('states', 'states_')
        if 'rates' in line:
            line = line.replace('rates', 'rates_')
        if 'variables' in line:
            line = line.replace('variables', 'variables_')
        if 'ExternalVariable externalVariable' in line:
            line = line.replace(', ExternalVariable externalVariable', '')
         
        postClassInit += line + '\n'
     

classInit += f"""
    voi = 0.0;
    states = createStatesArray();
    rates = createStatesArray(); // same size as states, should really be different function
    variables = createVariablesArray();
"""

if solver == 'RK4':
    classInit += """
    k1= createStatesArray(); // same size as states, should really be different function
    k2= createStatesArray(); // same size as states, should really be different function
    k3= createStatesArray(); // same size as states, should really be different function
    k4= createStatesArray(); // same size as states, should really be different function
    temp_states = createStatesArray();
    """

if solver == 'CVODE':
    # TODO do I have to calculate variables/states first?
    classInit += """
    // Create our SUNDIALS context.
    SUNContext_Create(NULL, &context);

    // Create our CVODE solver.
    solver = CVodeCreate(CV_BDF, context);

    // Initialise our CVODE solver.

    y = N_VMake_Serial(STATE_COUNT, states, context);

    CVodeInit(solver, func, voi, y);

    // Set our user data.

    userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this, std::placeholders::_1, 
                                                    std::placeholders::_2, std::placeholders::_3, 
                                                    std::placeholders::_4));

    CVodeSetUserData(solver, userData);

    // Set our maximum number of steps.

    CVodeSetMaxNumSteps(solver, 99999); // TODO get from user_inputs.yaml
    
    CVodeSetMaxStep(solver, 0.0001); // TODO get from user_inputs.yaml

    // Set our linear solver.

    matrix = SUNDenseMatrix(STATE_COUNT, STATE_COUNT, context);
    linearSolver = SUNLinSol_Dense(y, matrix, context);

    CVodeSetLinearSolver(solver, linearSolver, matrix);

    // Set our relative and absolute tolerances.

    CVodeSStolerances(solver, 1e-7, 1e-9); // TODO get from user_inputs.yaml
"""



# TODO the below need to be initialised in the member initialiser list
if len(variables_to_delay_info) > 0:
    # the below was in the classInit
    # EVSingleton* EVSingleton::instancePtr = NULL;
    # EVSingleton* evi = EVSingleton::getInstance();
    classInit += """
    // initialise buffers in the EVSingleton class
    evi->initBuffers(dt, variables); 

    """
classInit += """
}

Model0d::~Model0d() {
    // Clean up after ourselves.

    deleteArray(states);
    deleteArray(rates);
    deleteArray(variables);

"""
if solver == 'RK4':
    classInit += """
    deleteArray(k1);
    deleteArray(k2);
    deleteArray(k3);
    deleteArray(k4);
    deleteArray(temp_states);
"""
if solver == 'CVODE':
    classInit += """
    // Clean up after ourselves.

    SUNLinSolFree(linearSolver);
    SUNMatDestroy(matrix);
    N_VDestroy_Serial(y);
    CVodeFree(&solver);
    SUNContext_Free(&context);
"""
classInit += """
}
"""
# todo what else should I do in the destructor?

# and generate a function to compute external variables
computeEV = f"""
double Model0d::externalVariable(double voi, double *states, double *rates, double *variables, size_t index)
{{
  
"""
for ext_variable in external_variable_info:
    print(ext_variable.keys())

    # variable or state index
    if ext_variable["state_or_variable"] == "state":
        state_index = ext_variable['state_index']
    else:
        variable_index = ext_variable['variable_index']

    if ext_variable['variable_type'] == 'delay':
            
        delay_variable_index = ext_variable['delay_variable_index']
        delay_amount_index = ext_variable['delay_amount_index']
        computeEV += f'  if (index == {delay_variable_index}) {{\n'
        computeEV += f'    if (voi < variables[{delay_amount_index}]) {{\n'
        computeEV += f'      if (evi->isInitialised() != true) {{return 0.0;}};\n'
        if ext_variable["state_or_variable"] == "state":
            computeEV += f'      evi->storeVariable({delay_variable_index}, states[{state_index}]);\n'
        else:
            computeEV += f'      evi->storeVariable({delay_variable_index}, variables[{variable_index}]);\n'
        computeEV += f'      return 0.0;\n'
        computeEV += f'    }} else {{;\n'
        computeEV += f'    // save the current value of the variable to the circle buffer\n'
        if ext_variable["state_or_variable"] == "state":
            computeEV += f'      evi->storeVariable({delay_variable_index}, states[{state_index}]);\n'
        else:
            computeEV += f'      evi->storeVariable({delay_variable_index}, variables[{variable_index}]);\n'
        computeEV += f'      double value = evi->getVariable({delay_variable_index});\n'
        computeEV += f'      return value;\n'
        computeEV += f'    }};\n'
        computeEV += f'  }}\n'

    elif ext_variable['variable_type'] == 'input':
        if ext_variable['state_or_variable'] == 'state':
            print('Input BC variable can not be a state variable, exiting')
            exit()
            
        elif ext_variable['state_or_variable'] == 'variable':
            if ext_variable['coupled_to_type'] == 'vessel1d':
                computeEV += f'  if (index == {variable_index}) {{\n'
                if ext_variable['flow_or_pressure_bc'] == 'flow':
                    computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
                    computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
                    computeEV += f'    double value = get_model1d_flow(vessel1d_idx, inlet0_or_outlet1_bc);\n' 
                elif ext_variable['flow_or_pressure_bc'] == 'pressure':
                    computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"]\n'
                    computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"]\n'
                    computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, inlet0_or_outlet1_bc);\n' 
                computeEV += f'    return value;\n'
                computeEV += f'  }}\n'

computeEV += f"""
  return 0.0;
}}
"""

# external interaction functions
externalInteractionFunctions = """
void Model0d::connect_to_model1d(Model1d* model1d){
    model1d_ptr = model1d; 
}

double Model0d::get_model1d_flow(int vessel_idx, int input0_output1_bc){
    // get the flow from the 1d model
    double * all_vals = (double *) malloc(10*sizeof(double));
    if (input0_output1_bc == 0){
        model1d_ptr->evalSol(vessel_idx, 0.0, 0, all_vals); 
    } else {
        double xSample = model1d_ptr->vess[vessel_idx].L;
        int iSample = model1d_ptr->vess[vessel_idx].NCELLS - 1;
        model1d_ptr->evalSol(vessel_idx, xSample, iSample, all_vals); 
    }
    double flow = all_vals[1]/pow(10,6); // index 1 is the flow, divide by 10^6 to get m3/s from ml/s
    // TODO make the conversion of units more generic
    // std::cout << "flow: " << std::scientific << flow << std::endl;
    return flow; // index 1 is the flow, divide by 10^6 to get m3/s from ml/s
}

double Model0d::get_model1d_pressure(int vessel_idx, int input0_output1_bc){
    // get the flow from the 1d model
    double * all_vals = (double *) malloc(8*sizeof(double));
    if (input0_output1_bc == 0){
        model1d_ptr->sampleMid(vessel_idx, all_vals, 0.0); 
    } else {
        double xSample = model1d_ptr->vess[vessel_idx].L;
        model1d_ptr->sampleMid(vessel_idx, all_vals, xSample); 
    }
    return all_vals[4]; // index 4 is the pressure 
}

void Model0d::initialiseVariablesAndComputeConstants() {
    initialiseVariables(voi, states, rates, variables);
    computeComputedConstants(variables);
    computeRates(voi, states, rates, variables);
    computeVariables(voi, states, rates, variables);
"""
if solver == 'CVODE':
    externalInteractionFunctions += """
    // reinitialise the CVODE solver with the initialised state variables 
    y = N_VMake_Serial(STATE_COUNT, states, context);

    CVodeReInit(solver, voi, y);
"""

externalInteractionFunctions += """
    }
"""


# external interaction headers 
externalInteractionHeaders= """
    void connect_to_model1d(Model1d* model1d);
    double get_model1d_flow(int vessel_idx, int input0_output1_bc);
    double get_model1d_pressure(int vessel_idx, int input0_output1_bc);
    void initialiseVariablesAndComputeConstants();
"""

solveOneStepFunction = """
void Model0d::solveOneStep(double dt) {
"""
if solver == 'foreward_euler':
    solveOneStepFunction += """
    
    computeRates(voi, states, rates, variables);

    for (size_t i = 0; i < STATE_COUNT; ++i) {
        // simple forward Euler integration
        states[i] = states[i] + dt * rates[i];
    }
}
    """
elif solver == 'RK4':
    solveOneStepFunction += """
    // RK4 integration
    // first step: calculate k1
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i];
    }
    computeRates(voi, temp_states, k1, variables);

    // second step: calculate k2
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt/2.0 * k1[i];
    }
    computeRates(voi+dt/2.0, temp_states, k2, variables);
    
    // third step: calculate k3
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt/2.0 * k2[i];
    }
    computeRates(voi+dt/2.0, temp_states, k3, variables);
    
    // third step: calculate k4
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        temp_states[i] = states[i] + dt * k3[i];
    }
    computeRates(voi+dt, temp_states, k4, variables);

    for (size_t i = 0; i < STATE_COUNT; ++i) {
        rates[i] = 1.0/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        states[i] = states[i] + dt * rates[i];
    }
    computeVariables(voi, states, rates, variables);
    voi += dt;
}
    """

elif solver == 'CVODE':
    solveOneStepFunction += """ 

    voiEnd = voi + dt;
    // CVodeSetStopTime(solver, voiEnd);

    CVode(solver, voiEnd, y, &voi, CV_NORMAL);

    // Compute our variables.

    states = N_VGetArrayPointer_Serial(y);
    computeVariables(voiEnd, states, rates, variables);
    voi += dt;
}
    """
    

# typedef void (*computeRatesType)(double, double *, double *, double *);
userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates);

    double* variables() const;
    Model0d::FunctionType computeRates() const;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
};

"""

UserDataCC = """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates) :
    mVariables(pVariables),
    mComputeRates(pComputeRates)
{
}

//==============================================================================

double * UserOdeData::variables() const
{
    // Return our algebraic array

    return mVariables;
}

//==============================================================================

Model0d::FunctionType UserOdeData::computeRates() const
{
    // Return our compute rates function

    return mComputeRates;
}
"""

funcCC = """
int func(double voi, N_Vector y, N_Vector yDot, void *userData)
{
    UserOdeData *realUserData = static_cast<UserOdeData*>(userData);
    realUserData->computeRates()(voi, N_VGetArrayPointer_Serial(y), 
                               N_VGetArrayPointer_Serial(yDot), realUserData->variables());
    return 0;
}
    
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
    if ext_variable['variable_type'] == 'delay':
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

# TODO modify below with respect to circulatory_autogen inputs
# When coupling with a Cpp model that does the simulation, this isn't needed
mainScript = """
int main0d(void){
    double end_time = 5.0;
    double dt = 0.0000001;
    Model0d model0d_inst;
    double eps = 1e-12;

    while (model0d_inst.voi < end_time-eps) {
        model0d_inst.solveOneStep(dt);
    }

    // TODO autogenerate a dict of variable names to print with variables

    printf("Final values:");
    printf("  time: ");
    printf("%f", model0d_inst.voi);
    printf("  states:");
    for (size_t i = 0; i < model0d_inst.STATE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.states[i]);
    }
    printf("  variables:");
    for (size_t i = 0; i < model0d_inst.VARIABLE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.variables[i]);
    }

return 0;
}
"""

# save header to file
# with open(f'/home/farg967/software/venous_system/lucas_model/fvm/{model_name}.h', 'w') as f:
with open(f'/home/farg967/software/venous_system/lucas_model/fvm/model0d.h', 'w') as f:

    f.write(preHeaderStuff) 
    f.write(interFaceCodePreClass)
    f.write(classInitHeader)
    f.write(interFaceCodeInClass)
    f.write(externalInteractionHeaders)
    f.write(otherHeaderInits)
    f.write(classFinisherHeader)
    f.write(userDataHeader)

    if len(variables_to_delay_info) > 0:
        f.write(circularBufferHeader)
        f.write(storeEVSingletonHeader)

    
# and save implementation to file
# with open(f'/home/farg967/software/venous_system/lucas_model/fvm/{model_name}.cc', 'w') as f:
with open(f'/home/farg967/software/venous_system/lucas_model/fvm/model0d.cc', 'w') as f:

    f.write(preClassStuff)
    f.write(preSourceStuff) # this has to be below the #include "model.h" in implementationCode()
    f.write(funcCC)

    f.write(classInit)
    f.write(postClassInit)
    f.write(externalInteractionFunctions)
    f.write(computeEV)
    f.write(solveOneStepFunction)
    f.write(UserDataCC)
    
    if len(variables_to_delay_info) > 0:
        f.write(circularBuffer)
        f.write(storeEVSingleton)
    if create_main:
        f.write(mainScript)



