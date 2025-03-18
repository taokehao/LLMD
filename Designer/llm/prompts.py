
# This prompt is deprecated
REQUEST_TEMPLATE = """
I have a material and its ionic diffusion barrier value. The material is in the category of LLZO or \
variants of that. Ionic diffusion barrier is the minimum energy \
for the Lithium ion to move from one side to another inside the material. \
A lower diffusion barrier generally means higher ionic conductivity, \
which is important in applications such as batteries, fuel cells, and other electrochemical devices.

In the following text, I will provide you with the structure of the material and its ionic diffusion barrier.
The structure information will be given between symbols of ">>>" and "<<<". Each line represents an \
individual atom. On the left side of "|" is the element symbol of the atom and the right side is the \
(x,y,z) Cartesian coordinates of each atom.

Material Structure is: 
>>>
ATOMS_OF_THE_MATERIAL
<<<

Ionic diffusion barrier of this material is CURRENT_VALUE eV.


Please propose a modification to the material that achieves an ionic diffusion barrier value of TARGET_VALUE eV. \
You can choose one of the two following modifications:
1. exchange: exchange elements type of two atoms in the material while maintaining their position.
2. substitute: substitute the element type of an atom with another in the material while maintaining its position.

Your output should be a python dictionary of the following the format: \
{"Reason": $REASON, "Modification": [{"type": $TYPE, "action": ($ARG1, $ARG2)}, ...]} and do nothing else. \
Here are the requirements for your modifications:
1. $REASON should be your analysis and reason for choosing a modification
2. Value of Modification is a list of modifications, with each element representing a single modification \
given in the format of the dictionary provided above. The length of the list should be no more than MAX_MODIFICATIONS.
2. $TYPE should be the modification type; one of 'exchange', 'substitute'.
3. $ARG1 and $ARG2 should be the information needed for modification operations. For "exchange", \
$ARG1 and $ARG2 should be the indices of the two atoms to be exchanged, the index is same as line indices. The indices starts from 0. \
For "substitute", $ARG1 should be the index of the atom to be substituted same as the line indices, $ARG2 should be the element \
symbol of the new atom.
5. Modifications should not be performed to Lithium and nonmetal elements. And the candidate elements \
for substitution should be among transition metals.
6. Be careful about the ' and " in the sentence which may violate the Python syntax, you should properly use \\ to avoid that.
7. You can refer to the relevant contents of the knowledge base when modifying.
"""


REQUEST_TEMPLATE = """
I have a material categorized as LLZO or its variants, along with its ionic diffusion barrier value. The ionic diffusion barrier is the minimum energy required for a lithium ion to move through the material. A lower barrier generally indicates higher ionic conductivity, which is critical for applications like batteries, fuel cells, and other electrochemical devices.

Below, I will provide the material's structure and its ionic diffusion barrier value. The structure will be enclosed between ">>>" and "<<<", with each line representing an atom. The format is: Element Symbol | x, y, z where the coordinates are in Cartesian format.

Material Structure:
>>>
ATOMS_OF_THE_MATERIAL
<<<


Ionic diffusion Barrier: CURRENT_VALUE eV

Propose a modification to achieve a diffusion barrier of TARGET_VALUE eV. Use one of the following modification types:

1. exchange: Swap the element types of two atoms while keeping their positions.
2. substitute: Replace the element type of an atom with another while keeping its position.
Output the result as a Python dictionary in the format:
{
  "Reason": $REASON,
  "Modification": [
    {"type": $TYPE, "action": ($ARG1, $ARG2)},
    ...
  ]
}

Requirements:
1. Reason: Explain the analysis behind the chosen modifications.
2. Modification: A list of up to MAX_MODIFICATIONS, with each entry as described above.
3. $TYPE: Must be either "exchange" or "substitute".
4. $ARG1 and $ARG2:
    - For "exchange": $ARG1 and $ARG2 are the indices of the atoms to swap (indices correspond to line numbers, starting at 0).
    - For "substitute": $ARG1 is the atom index and $ARG2 is the new element symbol.
5. Do not modify Lithium or nonmetals. Substitutions should only use transition metals.
6. Ensure correct Python syntax, escaping quotes as needed.
7. You can refer to the relevant contents of the knowledge base when modifying.
"""





REQUEST_TEMPLATE_LOWER = """
I have a material and its ionic diffusion barrier value. The material is in the category of LLZO or \
variants of that. Ionic diffusion barrier is the minimum energy \
for the Lithium ion to move from one side to another inside the material. \
A lower diffusion barrier generally means higher ionic conductivity, \
which is important in applications such as batteries, fuel cells, and other electrochemical devices.

In the following text, I will provide you with the structure of the material and its ionic diffusion barrier.
The structure information will be given between symbols of ">>>" and "<<<". Each line represents an \
individual atom. On the left side of "|" is the element symbol of the atom and the right side is the \
(x,y,z) Cartesian coordinates of each atom.

Material Structure is: 
>>>
ATOMS_OF_THE_MATERIAL
<<<

Ionic diffusion barrier of this material is CURRENT_VALUE eV.


Please propose a modification to the material that achieves a lower ionic diffusion barrier value. \
You can choose one of the four following modifications:
1. exchange: exchange elements type of two atoms in the material while maintaining their position.
2. substitute: substitute the element type of an atom with another in the material while maintaining its position.

Your output should be a python dictionary of the following the format: \
{'Reason': $REASON, 'Modification': [{'type': $TYPE, 'action': ($ARG1, $ARG2)}, ...]}. \
Here are the requirements:
1. $REASON should be your analysis and reason for choosing a modification
2. Value of Modification is a list of modifications, with each element representing a single modification \
given in the format of the dictionary provided above. The length of the list should be no more than MAX_MODIFICATIONS.
2. $TYPE should be the modification type; one of 'exchange', 'substitute'.
3. $ARG1 and $ARG2 should be the information needed for modification operations. For "exchange", \
$ARG1 and $ARG2 should be the indices of the two atoms to be exchanged, the index is same as line indices. The indices starts from 0. \
For "substitute", $ARG1 should be the index of the atom to be substituted same as the line indices, $ARG2 should be the element \
symbol of the new atom.
5. Modifications should not be performed to Lithium and nonmetal elements. And the candidate elements \
for substitution should be among transition metals.
6. You can refer to the relevant contents of the knowledge base when modifying.
"""




REFLECTION_TEMPLATE = """
After performing modifications on previous material with a formula of PREVIOUS_FORMULA and \
produced a new material with a formula of NEW_FORMULA, the value of ionic diffusion barrier \
changed from PREVIOUS_VALUE eV to NEW_VALUE eV, while the target value is TARGET_VALUE eV. \
The modifications are suggested based on the reasons that REASON. 
Please write a brief post-action reflection on the modification in a short sentence about \
how successful the modifications were in achieving the target value of ionic diffusion \
barrier and why so.
In the following text, I will provide you with the structure of the materials. \
The structure information will be given between symbols of ">>>" and "<<<". \
Each line represents an individual atom. On the left side of "|" is the \
element symbol of the atom and the right side is the (x,y,z) Cartesian coordinates of each atom.

The origin structure is:
>>>
ATOMS_OF_THE_ORIGIN_MATERIAL
<<<

The newer structure is:
>>>
ATOMS_OF_THE_NEW_MATERIAL
<<<
"""


HISTORY_TEMPLATE = """
You can make use of the previous modifications and results below. In the following text the modifications are performed to the structure \
produced by the previous modifications. The sequence of the modifications is defined by the index in front of \
each line. Larger number means later modifications.
HISTORIES
"""

ONE_HISTORY_TEMPLATE = """
The modifications are: MODIFICATIONS, reasons for that is REASON. And after that modifications \
ionic diffusion barrier changed from PREVIOUS_VALUE to NEW_VALUE. \
Post-modification reflection is:  REFLECTION.
"""

