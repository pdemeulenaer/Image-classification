•щ$
Щэ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02unknown8уъ
v
cc3_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
Аƒ*
shared_namecc3_2/kernel
o
 cc3_2/kernel/Read/ReadVariableOpReadVariableOpcc3_2/kernel*
dtype0* 
_output_shapes
:
Аƒ
l

cc3_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
cc3_2/bias
e
cc3_2/bias/Read/ReadVariableOpReadVariableOp
cc3_2/bias*
dtype0*
_output_shapes
:
x
tags_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
Аƒ*
shared_nametags_2/kernel
q
!tags_2/kernel/Read/ReadVariableOpReadVariableOptags_2/kernel*
dtype0* 
_output_shapes
:
Аƒ
n
tags_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nametags_2/bias
g
tags_2/bias/Read/ReadVariableOpReadVariableOptags_2/bias*
dtype0*
_output_shapes
:
|
training_1/Adam/iterVarHandleOp*
shape: *%
shared_nametraining_1/Adam/iter*
dtype0	*
_output_shapes
: 
u
(training_1/Adam/iter/Read/ReadVariableOpReadVariableOptraining_1/Adam/iter*
dtype0	*
_output_shapes
: 
А
training_1/Adam/beta_1VarHandleOp*'
shared_nametraining_1/Adam/beta_1*
dtype0*
_output_shapes
: *
shape: 
y
*training_1/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_1/Adam/beta_1*
dtype0*
_output_shapes
: 
А
training_1/Adam/beta_2VarHandleOp*
shape: *'
shared_nametraining_1/Adam/beta_2*
dtype0*
_output_shapes
: 
y
*training_1/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_1/Adam/beta_2*
dtype0*
_output_shapes
: 
~
training_1/Adam/decayVarHandleOp*
shape: *&
shared_nametraining_1/Adam/decay*
dtype0*
_output_shapes
: 
w
)training_1/Adam/decay/Read/ReadVariableOpReadVariableOptraining_1/Adam/decay*
dtype0*
_output_shapes
: 
О
training_1/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *.
shared_nametraining_1/Adam/learning_rate
З
1training_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_1/Adam/learning_rate*
dtype0*
_output_shapes
: 
О
block1_conv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*&
shared_nameblock1_conv1_2/kernel
З
)block1_conv1_2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1_2/kernel*
dtype0*&
_output_shapes
:@
~
block1_conv1_2/biasVarHandleOp*$
shared_nameblock1_conv1_2/bias*
dtype0*
_output_shapes
: *
shape:@
w
'block1_conv1_2/bias/Read/ReadVariableOpReadVariableOpblock1_conv1_2/bias*
dtype0*
_output_shapes
:@
О
block1_conv2_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@@*&
shared_nameblock1_conv2_2/kernel
З
)block1_conv2_2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2_2/kernel*
dtype0*&
_output_shapes
:@@
~
block1_conv2_2/biasVarHandleOp*$
shared_nameblock1_conv2_2/bias*
dtype0*
_output_shapes
: *
shape:@
w
'block1_conv2_2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2_2/bias*
dtype0*
_output_shapes
:@
П
block2_conv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@А*&
shared_nameblock2_conv1_2/kernel
И
)block2_conv1_2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1_2/kernel*
dtype0*'
_output_shapes
:@А

block2_conv1_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock2_conv1_2/bias
x
'block2_conv1_2/bias/Read/ReadVariableOpReadVariableOpblock2_conv1_2/bias*
dtype0*
_output_shapes	
:А
Р
block2_conv2_2/kernelVarHandleOp*&
shared_nameblock2_conv2_2/kernel*
dtype0*
_output_shapes
: *
shape:АА
Й
)block2_conv2_2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2_2/kernel*
dtype0*(
_output_shapes
:АА

block2_conv2_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock2_conv2_2/bias
x
'block2_conv2_2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2_2/bias*
dtype0*
_output_shapes	
:А
Р
block3_conv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock3_conv1_2/kernel
Й
)block3_conv1_2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1_2/kernel*
dtype0*(
_output_shapes
:АА

block3_conv1_2/biasVarHandleOp*
shape:А*$
shared_nameblock3_conv1_2/bias*
dtype0*
_output_shapes
: 
x
'block3_conv1_2/bias/Read/ReadVariableOpReadVariableOpblock3_conv1_2/bias*
dtype0*
_output_shapes	
:А
Р
block3_conv2_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock3_conv2_2/kernel
Й
)block3_conv2_2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2_2/kernel*
dtype0*(
_output_shapes
:АА

block3_conv2_2/biasVarHandleOp*
shape:А*$
shared_nameblock3_conv2_2/bias*
dtype0*
_output_shapes
: 
x
'block3_conv2_2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2_2/bias*
dtype0*
_output_shapes	
:А
Р
block3_conv3_2/kernelVarHandleOp*&
shared_nameblock3_conv3_2/kernel*
dtype0*
_output_shapes
: *
shape:АА
Й
)block3_conv3_2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3_2/kernel*
dtype0*(
_output_shapes
:АА

block3_conv3_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock3_conv3_2/bias
x
'block3_conv3_2/bias/Read/ReadVariableOpReadVariableOpblock3_conv3_2/bias*
dtype0*
_output_shapes	
:А
Р
block3_conv4_2/kernelVarHandleOp*&
shared_nameblock3_conv4_2/kernel*
dtype0*
_output_shapes
: *
shape:АА
Й
)block3_conv4_2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4_2/kernel*
dtype0*(
_output_shapes
:АА

block3_conv4_2/biasVarHandleOp*
shape:А*$
shared_nameblock3_conv4_2/bias*
dtype0*
_output_shapes
: 
x
'block3_conv4_2/bias/Read/ReadVariableOpReadVariableOpblock3_conv4_2/bias*
dtype0*
_output_shapes	
:А
Р
block4_conv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock4_conv1_2/kernel
Й
)block4_conv1_2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1_2/kernel*
dtype0*(
_output_shapes
:АА

block4_conv1_2/biasVarHandleOp*
shape:А*$
shared_nameblock4_conv1_2/bias*
dtype0*
_output_shapes
: 
x
'block4_conv1_2/bias/Read/ReadVariableOpReadVariableOpblock4_conv1_2/bias*
dtype0*
_output_shapes	
:А
Р
block4_conv2_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock4_conv2_2/kernel
Й
)block4_conv2_2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2_2/kernel*
dtype0*(
_output_shapes
:АА

block4_conv2_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock4_conv2_2/bias
x
'block4_conv2_2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2_2/bias*
dtype0*
_output_shapes	
:А
Р
block4_conv3_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock4_conv3_2/kernel
Й
)block4_conv3_2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3_2/kernel*
dtype0*(
_output_shapes
:АА

block4_conv3_2/biasVarHandleOp*
shape:А*$
shared_nameblock4_conv3_2/bias*
dtype0*
_output_shapes
: 
x
'block4_conv3_2/bias/Read/ReadVariableOpReadVariableOpblock4_conv3_2/bias*
dtype0*
_output_shapes	
:А
Р
block4_conv4_2/kernelVarHandleOp*
shape:АА*&
shared_nameblock4_conv4_2/kernel*
dtype0*
_output_shapes
: 
Й
)block4_conv4_2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4_2/kernel*
dtype0*(
_output_shapes
:АА

block4_conv4_2/biasVarHandleOp*$
shared_nameblock4_conv4_2/bias*
dtype0*
_output_shapes
: *
shape:А
x
'block4_conv4_2/bias/Read/ReadVariableOpReadVariableOpblock4_conv4_2/bias*
dtype0*
_output_shapes	
:А
Р
block5_conv1_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock5_conv1_2/kernel
Й
)block5_conv1_2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1_2/kernel*
dtype0*(
_output_shapes
:АА

block5_conv1_2/biasVarHandleOp*$
shared_nameblock5_conv1_2/bias*
dtype0*
_output_shapes
: *
shape:А
x
'block5_conv1_2/bias/Read/ReadVariableOpReadVariableOpblock5_conv1_2/bias*
dtype0*
_output_shapes	
:А
Р
block5_conv2_2/kernelVarHandleOp*&
shared_nameblock5_conv2_2/kernel*
dtype0*
_output_shapes
: *
shape:АА
Й
)block5_conv2_2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2_2/kernel*
dtype0*(
_output_shapes
:АА

block5_conv2_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock5_conv2_2/bias
x
'block5_conv2_2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2_2/bias*
dtype0*
_output_shapes	
:А
Р
block5_conv3_2/kernelVarHandleOp*
shape:АА*&
shared_nameblock5_conv3_2/kernel*
dtype0*
_output_shapes
: 
Й
)block5_conv3_2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3_2/kernel*
dtype0*(
_output_shapes
:АА

block5_conv3_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameblock5_conv3_2/bias
x
'block5_conv3_2/bias/Read/ReadVariableOpReadVariableOpblock5_conv3_2/bias*
dtype0*
_output_shapes	
:А
Р
block5_conv4_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА*&
shared_nameblock5_conv4_2/kernel
Й
)block5_conv4_2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4_2/kernel*
dtype0*(
_output_shapes
:АА

block5_conv4_2/biasVarHandleOp*
shape:А*$
shared_nameblock5_conv4_2/bias*
dtype0*
_output_shapes
: 
x
'block5_conv4_2/bias/Read/ReadVariableOpReadVariableOpblock5_conv4_2/bias*
dtype0*
_output_shapes	
:А
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
b
total_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
b
count_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
dtype0*
_output_shapes
: 
Ъ
training_1/Adam/cc3_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
Аƒ*/
shared_name training_1/Adam/cc3_2/kernel/m
У
2training_1/Adam/cc3_2/kernel/m/Read/ReadVariableOpReadVariableOptraining_1/Adam/cc3_2/kernel/m*
dtype0* 
_output_shapes
:
Аƒ
Р
training_1/Adam/cc3_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*-
shared_nametraining_1/Adam/cc3_2/bias/m
Й
0training_1/Adam/cc3_2/bias/m/Read/ReadVariableOpReadVariableOptraining_1/Adam/cc3_2/bias/m*
dtype0*
_output_shapes
:
Ь
training_1/Adam/tags_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
Аƒ*0
shared_name!training_1/Adam/tags_2/kernel/m
Х
3training_1/Adam/tags_2/kernel/m/Read/ReadVariableOpReadVariableOptraining_1/Adam/tags_2/kernel/m*
dtype0* 
_output_shapes
:
Аƒ
Т
training_1/Adam/tags_2/bias/mVarHandleOp*.
shared_nametraining_1/Adam/tags_2/bias/m*
dtype0*
_output_shapes
: *
shape:
Л
1training_1/Adam/tags_2/bias/m/Read/ReadVariableOpReadVariableOptraining_1/Adam/tags_2/bias/m*
dtype0*
_output_shapes
:
Ъ
training_1/Adam/cc3_2/kernel/vVarHandleOp*
shape:
Аƒ*/
shared_name training_1/Adam/cc3_2/kernel/v*
dtype0*
_output_shapes
: 
У
2training_1/Adam/cc3_2/kernel/v/Read/ReadVariableOpReadVariableOptraining_1/Adam/cc3_2/kernel/v*
dtype0* 
_output_shapes
:
Аƒ
Р
training_1/Adam/cc3_2/bias/vVarHandleOp*
shape:*-
shared_nametraining_1/Adam/cc3_2/bias/v*
dtype0*
_output_shapes
: 
Й
0training_1/Adam/cc3_2/bias/v/Read/ReadVariableOpReadVariableOptraining_1/Adam/cc3_2/bias/v*
dtype0*
_output_shapes
:
Ь
training_1/Adam/tags_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
Аƒ*0
shared_name!training_1/Adam/tags_2/kernel/v
Х
3training_1/Adam/tags_2/kernel/v/Read/ReadVariableOpReadVariableOptraining_1/Adam/tags_2/kernel/v*
dtype0* 
_output_shapes
:
Аƒ
Т
training_1/Adam/tags_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_nametraining_1/Adam/tags_2/bias/v
Л
1training_1/Adam/tags_2/bias/v/Read/ReadVariableOpReadVariableOptraining_1/Adam/tags_2/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
хА
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *ѓА
value§АB†А BША
Н
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
Ґ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
 layer_with_weights-11
 layer-15
!layer-16
"layer_with_weights-12
"layer-17
#layer_with_weights-13
#layer-18
$layer_with_weights-14
$layer-19
%layer_with_weights-15
%layer-20
&layer-21
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
h

9kernel
:bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
Р
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate3m»4m…9m :mЋ3vћ4vЌ9vќ:vѕ
Ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31
332
433
934
:35
 

30
41
92
:3
Ъ
dmetrics

elayers
	variables
	regularization_losses
fnon_trainable_variables

trainable_variables
glayer_regularization_losses
 
 
 
 
Ъ
hmetrics

ilayers
	variables
regularization_losses
jnon_trainable_variables
trainable_variables
klayer_regularization_losses
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
h

Dkernel
Ebias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
h

Fkernel
Gbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
R
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
h

Hkernel
Ibias
|	variables
}regularization_losses
~trainable_variables
	keras_api
l

Jkernel
Kbias
А	variables
Бregularization_losses
Вtrainable_variables
Г	keras_api
V
Д	variables
Еregularization_losses
Жtrainable_variables
З	keras_api
l

Lkernel
Mbias
И	variables
Йregularization_losses
Кtrainable_variables
Л	keras_api
l

Nkernel
Obias
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
l

Pkernel
Qbias
Р	variables
Сregularization_losses
Тtrainable_variables
У	keras_api
l

Rkernel
Sbias
Ф	variables
Хregularization_losses
Цtrainable_variables
Ч	keras_api
V
Ш	variables
Щregularization_losses
Ъtrainable_variables
Ы	keras_api
l

Tkernel
Ubias
Ь	variables
Эregularization_losses
Юtrainable_variables
Я	keras_api
l

Vkernel
Wbias
†	variables
°regularization_losses
Ґtrainable_variables
£	keras_api
l

Xkernel
Ybias
§	variables
•regularization_losses
¶trainable_variables
І	keras_api
l

Zkernel
[bias
®	variables
©regularization_losses
™trainable_variables
Ђ	keras_api
V
ђ	variables
≠regularization_losses
Ѓtrainable_variables
ѓ	keras_api
l

\kernel
]bias
∞	variables
±regularization_losses
≤trainable_variables
≥	keras_api
l

^kernel
_bias
і	variables
µregularization_losses
ґtrainable_variables
Ј	keras_api
l

`kernel
abias
Є	variables
єregularization_losses
Їtrainable_variables
ї	keras_api
l

bkernel
cbias
Љ	variables
љregularization_losses
Њtrainable_variables
њ	keras_api
V
ј	variables
Ѕregularization_losses
¬trainable_variables
√	keras_api
ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31
 
 
Ю
ƒmetrics
≈layers
'	variables
(regularization_losses
∆non_trainable_variables
)trainable_variables
 «layer_regularization_losses
 
 
 
Ю
»metrics
…layers
+	variables
,regularization_losses
 non_trainable_variables
-trainable_variables
 Ћlayer_regularization_losses
 
 
 
Ю
ћmetrics
Ќlayers
/	variables
0regularization_losses
ќnon_trainable_variables
1trainable_variables
 ѕlayer_regularization_losses
XV
VARIABLE_VALUEcc3_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
cc3_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
Ю
–metrics
—layers
5	variables
6regularization_losses
“non_trainable_variables
7trainable_variables
 ”layer_regularization_losses
YW
VARIABLE_VALUEtags_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtags_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
Ю
‘metrics
’layers
;	variables
<regularization_losses
÷non_trainable_variables
=trainable_variables
 „layer_regularization_losses
SQ
VARIABLE_VALUEtraining_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblock1_conv1_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv1_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblock1_conv2_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblock2_conv1_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblock2_conv2_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblock3_conv1_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock3_conv2_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock3_conv3_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock3_conv4_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv4_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock4_conv1_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock4_conv2_2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2_2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock4_conv3_2/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3_2/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock4_conv4_2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv4_2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock5_conv1_2/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1_2/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock5_conv2_2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2_2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock5_conv3_2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3_2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblock5_conv4_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv4_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE

Ў0
ў1
*
0
1
2
3
4
5
ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31
 
 
 
 
 
 
 
 
Ю
Џmetrics
џlayers
l	variables
mregularization_losses
№non_trainable_variables
ntrainable_variables
 Ёlayer_regularization_losses

D0
E1
 
 
Ю
ёmetrics
яlayers
p	variables
qregularization_losses
аnon_trainable_variables
rtrainable_variables
 бlayer_regularization_losses

F0
G1
 
 
Ю
вmetrics
гlayers
t	variables
uregularization_losses
дnon_trainable_variables
vtrainable_variables
 еlayer_regularization_losses
 
 
 
Ю
жmetrics
зlayers
x	variables
yregularization_losses
иnon_trainable_variables
ztrainable_variables
 йlayer_regularization_losses

H0
I1
 
 
Ю
кmetrics
лlayers
|	variables
}regularization_losses
мnon_trainable_variables
~trainable_variables
 нlayer_regularization_losses

J0
K1
 
 
°
оmetrics
пlayers
А	variables
Бregularization_losses
рnon_trainable_variables
Вtrainable_variables
 сlayer_regularization_losses
 
 
 
°
тmetrics
уlayers
Д	variables
Еregularization_losses
фnon_trainable_variables
Жtrainable_variables
 хlayer_regularization_losses

L0
M1
 
 
°
цmetrics
чlayers
И	variables
Йregularization_losses
шnon_trainable_variables
Кtrainable_variables
 щlayer_regularization_losses

N0
O1
 
 
°
ъmetrics
ыlayers
М	variables
Нregularization_losses
ьnon_trainable_variables
Оtrainable_variables
 эlayer_regularization_losses

P0
Q1
 
 
°
юmetrics
€layers
Р	variables
Сregularization_losses
Аnon_trainable_variables
Тtrainable_variables
 Бlayer_regularization_losses

R0
S1
 
 
°
Вmetrics
Гlayers
Ф	variables
Хregularization_losses
Дnon_trainable_variables
Цtrainable_variables
 Еlayer_regularization_losses
 
 
 
°
Жmetrics
Зlayers
Ш	variables
Щregularization_losses
Иnon_trainable_variables
Ъtrainable_variables
 Йlayer_regularization_losses

T0
U1
 
 
°
Кmetrics
Лlayers
Ь	variables
Эregularization_losses
Мnon_trainable_variables
Юtrainable_variables
 Нlayer_regularization_losses

V0
W1
 
 
°
Оmetrics
Пlayers
†	variables
°regularization_losses
Рnon_trainable_variables
Ґtrainable_variables
 Сlayer_regularization_losses

X0
Y1
 
 
°
Тmetrics
Уlayers
§	variables
•regularization_losses
Фnon_trainable_variables
¶trainable_variables
 Хlayer_regularization_losses

Z0
[1
 
 
°
Цmetrics
Чlayers
®	variables
©regularization_losses
Шnon_trainable_variables
™trainable_variables
 Щlayer_regularization_losses
 
 
 
°
Ъmetrics
Ыlayers
ђ	variables
≠regularization_losses
Ьnon_trainable_variables
Ѓtrainable_variables
 Эlayer_regularization_losses

\0
]1
 
 
°
Юmetrics
Яlayers
∞	variables
±regularization_losses
†non_trainable_variables
≤trainable_variables
 °layer_regularization_losses

^0
_1
 
 
°
Ґmetrics
£layers
і	variables
µregularization_losses
§non_trainable_variables
ґtrainable_variables
 •layer_regularization_losses

`0
a1
 
 
°
¶metrics
Іlayers
Є	variables
єregularization_losses
®non_trainable_variables
Їtrainable_variables
 ©layer_regularization_losses

b0
c1
 
 
°
™metrics
Ђlayers
Љ	variables
љregularization_losses
ђnon_trainable_variables
Њtrainable_variables
 ≠layer_regularization_losses
 
 
 
°
Ѓmetrics
ѓlayers
ј	variables
Ѕregularization_losses
∞non_trainable_variables
¬trainable_variables
 ±layer_regularization_losses
 
¶
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


≤total

≥count
і
_fn_kwargs
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api


єtotal

Їcount
ї
_fn_kwargs
Љ	variables
љregularization_losses
Њtrainable_variables
њ	keras_api
 
 
 
 
 
 

D0
E1
 
 
 

F0
G1
 
 
 
 
 
 
 

H0
I1
 
 
 

J0
K1
 
 
 
 
 
 
 

L0
M1
 
 
 

N0
O1
 
 
 

P0
Q1
 
 
 

R0
S1
 
 
 
 
 
 
 

T0
U1
 
 
 

V0
W1
 
 
 

X0
Y1
 
 
 

Z0
[1
 
 
 
 
 
 
 

\0
]1
 
 
 

^0
_1
 
 
 

`0
a1
 
 
 

b0
c1
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

≤0
≥1
 
 
°
јmetrics
Ѕlayers
µ	variables
ґregularization_losses
¬non_trainable_variables
Јtrainable_variables
 √layer_regularization_losses
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

є0
Ї1
 
 
°
ƒmetrics
≈layers
Љ	variables
љregularization_losses
∆non_trainable_variables
Њtrainable_variables
 «layer_regularization_losses
 
 

≤0
≥1
 
 
 

є0
Ї1
 
ЗД
VARIABLE_VALUEtraining_1/Adam/cc3_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEtraining_1/Adam/cc3_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUEtraining_1/Adam/tags_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEtraining_1/Adam/tags_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUEtraining_1/Adam/cc3_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEtraining_1/Adam/cc3_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUEtraining_1/Adam/tags_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEtraining_1/Adam/tags_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
М
serving_default_inputPlaceholder*
dtype0*1
_output_shapes
:€€€€€€€€€аа*&
shape:€€€€€€€€€аа
—
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputblock1_conv1_2/kernelblock1_conv1_2/biasblock1_conv2_2/kernelblock1_conv2_2/biasblock2_conv1_2/kernelblock2_conv1_2/biasblock2_conv2_2/kernelblock2_conv2_2/biasblock3_conv1_2/kernelblock3_conv1_2/biasblock3_conv2_2/kernelblock3_conv2_2/biasblock3_conv3_2/kernelblock3_conv3_2/biasblock3_conv4_2/kernelblock3_conv4_2/biasblock4_conv1_2/kernelblock4_conv1_2/biasblock4_conv2_2/kernelblock4_conv2_2/biasblock4_conv3_2/kernelblock4_conv3_2/biasblock4_conv4_2/kernelblock4_conv4_2/biasblock5_conv1_2/kernelblock5_conv1_2/biasblock5_conv2_2/kernelblock5_conv2_2/biasblock5_conv3_2/kernelblock5_conv3_2/biasblock5_conv4_2/kernelblock5_conv4_2/biastags_2/kerneltags_2/biascc3_2/kernel
cc3_2/bias*-
_gradient_op_typePartitionedCall-275993*-
f(R&
$__inference_signature_wrapper_274780*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*0
Tin)
'2%
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
ч
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename cc3_2/kernel/Read/ReadVariableOpcc3_2/bias/Read/ReadVariableOp!tags_2/kernel/Read/ReadVariableOptags_2/bias/Read/ReadVariableOp(training_1/Adam/iter/Read/ReadVariableOp*training_1/Adam/beta_1/Read/ReadVariableOp*training_1/Adam/beta_2/Read/ReadVariableOp)training_1/Adam/decay/Read/ReadVariableOp1training_1/Adam/learning_rate/Read/ReadVariableOp)block1_conv1_2/kernel/Read/ReadVariableOp'block1_conv1_2/bias/Read/ReadVariableOp)block1_conv2_2/kernel/Read/ReadVariableOp'block1_conv2_2/bias/Read/ReadVariableOp)block2_conv1_2/kernel/Read/ReadVariableOp'block2_conv1_2/bias/Read/ReadVariableOp)block2_conv2_2/kernel/Read/ReadVariableOp'block2_conv2_2/bias/Read/ReadVariableOp)block3_conv1_2/kernel/Read/ReadVariableOp'block3_conv1_2/bias/Read/ReadVariableOp)block3_conv2_2/kernel/Read/ReadVariableOp'block3_conv2_2/bias/Read/ReadVariableOp)block3_conv3_2/kernel/Read/ReadVariableOp'block3_conv3_2/bias/Read/ReadVariableOp)block3_conv4_2/kernel/Read/ReadVariableOp'block3_conv4_2/bias/Read/ReadVariableOp)block4_conv1_2/kernel/Read/ReadVariableOp'block4_conv1_2/bias/Read/ReadVariableOp)block4_conv2_2/kernel/Read/ReadVariableOp'block4_conv2_2/bias/Read/ReadVariableOp)block4_conv3_2/kernel/Read/ReadVariableOp'block4_conv3_2/bias/Read/ReadVariableOp)block4_conv4_2/kernel/Read/ReadVariableOp'block4_conv4_2/bias/Read/ReadVariableOp)block5_conv1_2/kernel/Read/ReadVariableOp'block5_conv1_2/bias/Read/ReadVariableOp)block5_conv2_2/kernel/Read/ReadVariableOp'block5_conv2_2/bias/Read/ReadVariableOp)block5_conv3_2/kernel/Read/ReadVariableOp'block5_conv3_2/bias/Read/ReadVariableOp)block5_conv4_2/kernel/Read/ReadVariableOp'block5_conv4_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp2training_1/Adam/cc3_2/kernel/m/Read/ReadVariableOp0training_1/Adam/cc3_2/bias/m/Read/ReadVariableOp3training_1/Adam/tags_2/kernel/m/Read/ReadVariableOp1training_1/Adam/tags_2/bias/m/Read/ReadVariableOp2training_1/Adam/cc3_2/kernel/v/Read/ReadVariableOp0training_1/Adam/cc3_2/bias/v/Read/ReadVariableOp3training_1/Adam/tags_2/kernel/v/Read/ReadVariableOp1training_1/Adam/tags_2/bias/v/Read/ReadVariableOpConst*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
_output_shapes
: *B
Tin;
927	*-
_gradient_op_typePartitionedCall-276069*(
f#R!
__inference__traced_save_276068*
Tout
2
ќ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecc3_2/kernel
cc3_2/biastags_2/kerneltags_2/biastraining_1/Adam/itertraining_1/Adam/beta_1training_1/Adam/beta_2training_1/Adam/decaytraining_1/Adam/learning_rateblock1_conv1_2/kernelblock1_conv1_2/biasblock1_conv2_2/kernelblock1_conv2_2/biasblock2_conv1_2/kernelblock2_conv1_2/biasblock2_conv2_2/kernelblock2_conv2_2/biasblock3_conv1_2/kernelblock3_conv1_2/biasblock3_conv2_2/kernelblock3_conv2_2/biasblock3_conv3_2/kernelblock3_conv3_2/biasblock3_conv4_2/kernelblock3_conv4_2/biasblock4_conv1_2/kernelblock4_conv1_2/biasblock4_conv2_2/kernelblock4_conv2_2/biasblock4_conv3_2/kernelblock4_conv3_2/biasblock4_conv4_2/kernelblock4_conv4_2/biasblock5_conv1_2/kernelblock5_conv1_2/biasblock5_conv2_2/kernelblock5_conv2_2/biasblock5_conv3_2/kernelblock5_conv3_2/biasblock5_conv4_2/kernelblock5_conv4_2/biastotalcounttotal_1count_1training_1/Adam/cc3_2/kernel/mtraining_1/Adam/cc3_2/bias/mtraining_1/Adam/tags_2/kernel/mtraining_1/Adam/tags_2/bias/mtraining_1/Adam/cc3_2/kernel/vtraining_1/Adam/cc3_2/bias/vtraining_1/Adam/tags_2/kernel/vtraining_1/Adam/tags_2/bias/v*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*A
Tin:
826*
_output_shapes
: *-
_gradient_op_typePartitionedCall-276241*+
f&R$
"__inference__traced_restore_276240ьв
э
_
C__inference_flatten_layer_call_and_return_conditional_losses_274373

inputs
identity^
Reshape/shapeConst*
valueB"€€€€ b  *
dtype0*
_output_shapes
:f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒZ
IdentityIdentityReshape:output:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
√1
и
A__inference_model_layer_call_and_return_conditional_losses_274591

inputs(
$vgg19_statefulpartitionedcall_args_1(
$vgg19_statefulpartitionedcall_args_2(
$vgg19_statefulpartitionedcall_args_3(
$vgg19_statefulpartitionedcall_args_4(
$vgg19_statefulpartitionedcall_args_5(
$vgg19_statefulpartitionedcall_args_6(
$vgg19_statefulpartitionedcall_args_7(
$vgg19_statefulpartitionedcall_args_8(
$vgg19_statefulpartitionedcall_args_9)
%vgg19_statefulpartitionedcall_args_10)
%vgg19_statefulpartitionedcall_args_11)
%vgg19_statefulpartitionedcall_args_12)
%vgg19_statefulpartitionedcall_args_13)
%vgg19_statefulpartitionedcall_args_14)
%vgg19_statefulpartitionedcall_args_15)
%vgg19_statefulpartitionedcall_args_16)
%vgg19_statefulpartitionedcall_args_17)
%vgg19_statefulpartitionedcall_args_18)
%vgg19_statefulpartitionedcall_args_19)
%vgg19_statefulpartitionedcall_args_20)
%vgg19_statefulpartitionedcall_args_21)
%vgg19_statefulpartitionedcall_args_22)
%vgg19_statefulpartitionedcall_args_23)
%vgg19_statefulpartitionedcall_args_24)
%vgg19_statefulpartitionedcall_args_25)
%vgg19_statefulpartitionedcall_args_26)
%vgg19_statefulpartitionedcall_args_27)
%vgg19_statefulpartitionedcall_args_28)
%vgg19_statefulpartitionedcall_args_29)
%vgg19_statefulpartitionedcall_args_30)
%vgg19_statefulpartitionedcall_args_31)
%vgg19_statefulpartitionedcall_args_32'
#tags_statefulpartitionedcall_args_1'
#tags_statefulpartitionedcall_args_2&
"cc3_statefulpartitionedcall_args_1&
"cc3_statefulpartitionedcall_args_2
identity

identity_1ИҐcc3/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐtags/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallЈ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputs$vgg19_statefulpartitionedcall_args_1$vgg19_statefulpartitionedcall_args_2$vgg19_statefulpartitionedcall_args_3$vgg19_statefulpartitionedcall_args_4$vgg19_statefulpartitionedcall_args_5$vgg19_statefulpartitionedcall_args_6$vgg19_statefulpartitionedcall_args_7$vgg19_statefulpartitionedcall_args_8$vgg19_statefulpartitionedcall_args_9%vgg19_statefulpartitionedcall_args_10%vgg19_statefulpartitionedcall_args_11%vgg19_statefulpartitionedcall_args_12%vgg19_statefulpartitionedcall_args_13%vgg19_statefulpartitionedcall_args_14%vgg19_statefulpartitionedcall_args_15%vgg19_statefulpartitionedcall_args_16%vgg19_statefulpartitionedcall_args_17%vgg19_statefulpartitionedcall_args_18%vgg19_statefulpartitionedcall_args_19%vgg19_statefulpartitionedcall_args_20%vgg19_statefulpartitionedcall_args_21%vgg19_statefulpartitionedcall_args_22%vgg19_statefulpartitionedcall_args_23%vgg19_statefulpartitionedcall_args_24%vgg19_statefulpartitionedcall_args_25%vgg19_statefulpartitionedcall_args_26%vgg19_statefulpartitionedcall_args_27%vgg19_statefulpartitionedcall_args_28%vgg19_statefulpartitionedcall_args_29%vgg19_statefulpartitionedcall_args_30%vgg19_statefulpartitionedcall_args_31%vgg19_statefulpartitionedcall_args_32*-
_gradient_op_typePartitionedCall-274260*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274135*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€АЌ
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*)
_output_shapes
:€€€€€€€€€Аƒ*-
_gradient_op_typePartitionedCall-274379*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_274373*
Tout
2„
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274406*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2*-
_gradient_op_typePartitionedCall-274417~
	tags/CastCast(dropout/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒИ
tags/StatefulPartitionedCallStatefulPartitionedCalltags/Cast:y:0#tags_statefulpartitionedcall_args_1#tags_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-274448*I
fDRB
@__inference_tags_layer_call_and_return_conditional_losses_274442*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*'
_output_shapes
:€€€€€€€€€*
Tin
2}
cc3/CastCast(dropout/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒГ
cc3/StatefulPartitionedCallStatefulPartitionedCallcc3/Cast:y:0"cc3_statefulpartitionedcall_args_1"cc3_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-274477*H
fCRA
?__inference_cc3_layer_call_and_return_conditional_losses_274471л
IdentityIdentity$cc3/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€о

Identity_1Identity%tags/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"!

identity_1Identity_1:output:0"
identityIdentity:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2<
tags/StatefulPartitionedCalltags/StatefulPartitionedCall2:
cc3/StatefulPartitionedCallcc3/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ 
ѓ
H
,__inference_block4_pool_layer_call_fn_273581

inputs
identity 
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-273578*P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Љп
и
A__inference_model_layer_call_and_return_conditional_losses_274939

inputs5
1vgg19_block1_conv1_conv2d_readvariableop_resource6
2vgg19_block1_conv1_biasadd_readvariableop_resource5
1vgg19_block1_conv2_conv2d_readvariableop_resource6
2vgg19_block1_conv2_biasadd_readvariableop_resource5
1vgg19_block2_conv1_conv2d_readvariableop_resource6
2vgg19_block2_conv1_biasadd_readvariableop_resource5
1vgg19_block2_conv2_conv2d_readvariableop_resource6
2vgg19_block2_conv2_biasadd_readvariableop_resource5
1vgg19_block3_conv1_conv2d_readvariableop_resource6
2vgg19_block3_conv1_biasadd_readvariableop_resource5
1vgg19_block3_conv2_conv2d_readvariableop_resource6
2vgg19_block3_conv2_biasadd_readvariableop_resource5
1vgg19_block3_conv3_conv2d_readvariableop_resource6
2vgg19_block3_conv3_biasadd_readvariableop_resource5
1vgg19_block3_conv4_conv2d_readvariableop_resource6
2vgg19_block3_conv4_biasadd_readvariableop_resource5
1vgg19_block4_conv1_conv2d_readvariableop_resource6
2vgg19_block4_conv1_biasadd_readvariableop_resource5
1vgg19_block4_conv2_conv2d_readvariableop_resource6
2vgg19_block4_conv2_biasadd_readvariableop_resource5
1vgg19_block4_conv3_conv2d_readvariableop_resource6
2vgg19_block4_conv3_biasadd_readvariableop_resource5
1vgg19_block4_conv4_conv2d_readvariableop_resource6
2vgg19_block4_conv4_biasadd_readvariableop_resource5
1vgg19_block5_conv1_conv2d_readvariableop_resource6
2vgg19_block5_conv1_biasadd_readvariableop_resource5
1vgg19_block5_conv2_conv2d_readvariableop_resource6
2vgg19_block5_conv2_biasadd_readvariableop_resource5
1vgg19_block5_conv3_conv2d_readvariableop_resource6
2vgg19_block5_conv3_biasadd_readvariableop_resource5
1vgg19_block5_conv4_conv2d_readvariableop_resource6
2vgg19_block5_conv4_biasadd_readvariableop_resource'
#tags_matmul_readvariableop_resource(
$tags_biasadd_readvariableop_resource&
"cc3_matmul_readvariableop_resource'
#cc3_biasadd_readvariableop_resource
identity

identity_1ИҐcc3/BiasAdd/ReadVariableOpҐcc3/MatMul/ReadVariableOpҐtags/BiasAdd/ReadVariableOpҐtags/MatMul/ReadVariableOpҐ)vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv1/Conv2D/ReadVariableOpҐ)vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv2/Conv2D/ReadVariableOpҐ)vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv1/Conv2D/ReadVariableOpҐ)vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv1/Conv2D/ReadVariableOpҐ)vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv3/Conv2D/ReadVariableOpҐ)vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv4/Conv2D/ReadVariableOpҐ)vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv1/Conv2D/ReadVariableOpҐ)vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv2/Conv2D/ReadVariableOpҐ)vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv3/Conv2D/ReadVariableOpҐ)vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv4/Conv2D/ReadVariableOpҐ)vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv1/Conv2D/ReadVariableOpҐ)vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv2/Conv2D/ReadVariableOpҐ)vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv3/Conv2D/ReadVariableOpҐ)vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv4/Conv2D/ReadVariableOpr
vgg19/block1_conv1/CastCastinputs*

DstT0*1
_output_shapes
:€€€€€€€€€аа*

SrcT0–
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@÷
vgg19/block1_conv1/Conv2DConv2Dvgg19/block1_conv1/Cast:y:00vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@∆
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Є
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0А
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@–
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@а
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
*
paddingSAME∆
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Є
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0А
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€аа@*
T0Ѕ
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:€€€€€€€€€pp@—
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А№
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
*
paddingSAME«
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0“
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА«
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0¬
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А“
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAME«
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А“
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0«
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0“
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
«
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0“
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAME«
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А¬
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0«
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0«
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0“
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAME«
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А¬
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALID“
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0“
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
«
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0“
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А¬
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€Аf
flatten/Reshape/shapeConst*
valueB"€€€€ b  *
dtype0*
_output_shapes
:Т
flatten/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten/Reshape/shape:output:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0W
dropout/dropout/rateConst*
value
B jАh*
dtype0*
_output_shapes
: ]
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:d
"dropout/dropout/random_uniform/minConst*
value	B j *
dtype0*
_output_shapes
: e
"dropout/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
value
B jАxЮ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*)
_output_shapes
:€€€€€€€€€Аƒ§
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Љ
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*)
_output_shapes
:€€€€€€€€€АƒЃ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒX
dropout/dropout/sub/xConst*
dtype0*
_output_shapes
: *
value
B jАxz
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0\
dropout/dropout/truediv/xConst*
value
B jАx*
dtype0*
_output_shapes
: А
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: £
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒЕ
dropout/dropout/mulMulflatten/Reshape:output:0dropout/dropout/truediv:z:0*
T0*)
_output_shapes
:€€€€€€€€€АƒБ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*)
_output_shapes
:€€€€€€€€€АƒГ
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒo
	tags/CastCastdropout/dropout/mul_1:z:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒЃ
tags/MatMul/ReadVariableOpReadVariableOp#tags_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒz
tags/MatMulMatMultags/Cast:y:0"tags/MatMul/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0™
tags/BiasAdd/ReadVariableOpReadVariableOp$tags_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Е
tags/BiasAddBiasAddtags/MatMul:product:0#tags/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0`
tags/SigmoidSigmoidtags/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
cc3/CastCastdropout/dropout/mul_1:z:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€Аƒђ
cc3/MatMul/ReadVariableOpReadVariableOp"cc3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒw

cc3/MatMulMatMulcc3/Cast:y:0!cc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€®
cc3/BiasAdd/ReadVariableOpReadVariableOp#cc3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:В
cc3/BiasAddBiasAddcc3/MatMul:product:0"cc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
cc3/SoftmaxSoftmaxcc3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ѕ
IdentityIdentitycc3/Softmax:softmax:0^cc3/BiasAdd/ReadVariableOp^cc3/MatMul/ReadVariableOp^tags/BiasAdd/ReadVariableOp^tags/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Њ

Identity_1Identitytags/Sigmoid:y:0^cc3/BiasAdd/ReadVariableOp^cc3/MatMul/ReadVariableOp^tags/BiasAdd/ReadVariableOp^tags/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp28
tags/MatMul/ReadVariableOptags/MatMul/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp28
cc3/BiasAdd/ReadVariableOpcc3/BiasAdd/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp26
cc3/MatMul/ReadVariableOpcc3/MatMul/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2:
tags/BiasAdd/ReadVariableOptags/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp:$ :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# 
Ї
Ѓ
-__inference_block2_conv1_layer_call_fn_273305

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273300*Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ѓ
H
,__inference_block5_pool_layer_call_fn_273698

inputs
identity 
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-273695*P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
и
Щ
&__inference_model_layer_call_fn_274727	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36
identity

identity_1ИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
Tin)
'2%*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*-
_gradient_op_typePartitionedCall-274686*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_274685В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€Д

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"!

identity_1Identity_1:output:0"
identityIdentity:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ 
П
б
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Й
a
C__inference_dropout_layer_call_and_return_conditional_losses_274413

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:€€€€€€€€€Аƒ]

Identity_1IdentityIdentity:output:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0"!

identity_1Identity_1:output:0*(
_input_shapes
:€€€€€€€€€Аƒ:& "
 
_user_specified_nameinputs
µђ
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_275288

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0Ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0Ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
*
ksize
*
paddingVALID≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
*
paddingSAMEї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
ksize
*
paddingVALID∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAMEї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAMEї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0Э

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
≤≠
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_275727

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpl
block1_conv1/CastCastinputs*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€ааƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ƒ
block1_conv1/Conv2DConv2Dblock1_conv1/Cast:y:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0Ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0Ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppАї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAMEї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0ґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALID∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0Э

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
П
б
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ґ
c
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Ґ
c
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ѓ
H
,__inference_block1_pool_layer_call_fn_273280

inputs
identity 
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-273277*P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tin
2Г
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
Й
a
C__inference_dropout_layer_call_and_return_conditional_losses_275837

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:€€€€€€€€€Аƒ]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"!

identity_1Identity_1:output:0*(
_input_shapes
:€€€€€€€€€Аƒ:& "
 
_user_specified_nameinputs
л
ц	
&__inference_vgg19_layer_call_fn_275801

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallс

StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274257*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*,
Tin%
#2!*-
_gradient_op_typePartitionedCall-274298Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
ї
Ѓ
-__inference_block4_conv3_layer_call_fn_273539

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273534*Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
√
D
(__inference_dropout_layer_call_fn_275847

inputs
identity•
PartitionedCallPartitionedCallinputs*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2*-
_gradient_op_typePartitionedCall-274425*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274413*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄb
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"
identityIdentity:output:0*(
_input_shapes
:€€€€€€€€€Аƒ:& "
 
_user_specified_nameinputs
оl
с
A__inference_vgg19_layer_call_and_return_conditional_losses_273817
input_1/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block3_conv4_statefulpartitionedcall_args_1/
+block3_conv4_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block4_conv4_statefulpartitionedcall_args_1/
+block4_conv4_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2/
+block5_conv4_statefulpartitionedcall_args_1/
+block5_conv4_statefulpartitionedcall_args_2
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCallђ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*1
_output_shapes
:€€€€€€€€€аа@*-
_gradient_op_typePartitionedCall-273233*Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227“
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*1
_output_shapes
:€€€€€€€€€аа@*
Tin
2*-
_gradient_op_typePartitionedCall-273258*Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252*
Tout
2в
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273277*P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*/
_output_shapes
:€€€€€€€€€pp@»
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273300*Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€ppА*
Tin
2—
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€ppА*-
_gradient_op_typePartitionedCall-273325г
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273344*P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€88А*
Tin
2»
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€88А*
Tin
2*-
_gradient_op_typePartitionedCall-273367*Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361*
Tout
2—
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€88А*
Tin
2*-
_gradient_op_typePartitionedCall-273392*Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386—
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273417*Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411*
Tout
2—
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0+block3_conv4_statefulpartitionedcall_args_1+block3_conv4_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273442*Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄг
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273461*P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А»
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273484*Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2—
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273509*Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А—
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273534*Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0+block4_conv4_statefulpartitionedcall_args_1+block4_conv4_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273559*Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553г
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273578*P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2»
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273601*Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595—
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273626*Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620*
Tout
2—
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273651*Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645*
Tout
2—
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0+block5_conv4_statefulpartitionedcall_args_1+block5_conv4_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273676*Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670*
Tout
2г
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273695*P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€Ае
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
П
б
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ц/
≈
A__inference_model_layer_call_and_return_conditional_losses_274540	
input(
$vgg19_statefulpartitionedcall_args_1(
$vgg19_statefulpartitionedcall_args_2(
$vgg19_statefulpartitionedcall_args_3(
$vgg19_statefulpartitionedcall_args_4(
$vgg19_statefulpartitionedcall_args_5(
$vgg19_statefulpartitionedcall_args_6(
$vgg19_statefulpartitionedcall_args_7(
$vgg19_statefulpartitionedcall_args_8(
$vgg19_statefulpartitionedcall_args_9)
%vgg19_statefulpartitionedcall_args_10)
%vgg19_statefulpartitionedcall_args_11)
%vgg19_statefulpartitionedcall_args_12)
%vgg19_statefulpartitionedcall_args_13)
%vgg19_statefulpartitionedcall_args_14)
%vgg19_statefulpartitionedcall_args_15)
%vgg19_statefulpartitionedcall_args_16)
%vgg19_statefulpartitionedcall_args_17)
%vgg19_statefulpartitionedcall_args_18)
%vgg19_statefulpartitionedcall_args_19)
%vgg19_statefulpartitionedcall_args_20)
%vgg19_statefulpartitionedcall_args_21)
%vgg19_statefulpartitionedcall_args_22)
%vgg19_statefulpartitionedcall_args_23)
%vgg19_statefulpartitionedcall_args_24)
%vgg19_statefulpartitionedcall_args_25)
%vgg19_statefulpartitionedcall_args_26)
%vgg19_statefulpartitionedcall_args_27)
%vgg19_statefulpartitionedcall_args_28)
%vgg19_statefulpartitionedcall_args_29)
%vgg19_statefulpartitionedcall_args_30)
%vgg19_statefulpartitionedcall_args_31)
%vgg19_statefulpartitionedcall_args_32'
#tags_statefulpartitionedcall_args_1'
#tags_statefulpartitionedcall_args_2&
"cc3_statefulpartitionedcall_args_1&
"cc3_statefulpartitionedcall_args_2
identity

identity_1ИҐcc3/StatefulPartitionedCallҐtags/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallґ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput$vgg19_statefulpartitionedcall_args_1$vgg19_statefulpartitionedcall_args_2$vgg19_statefulpartitionedcall_args_3$vgg19_statefulpartitionedcall_args_4$vgg19_statefulpartitionedcall_args_5$vgg19_statefulpartitionedcall_args_6$vgg19_statefulpartitionedcall_args_7$vgg19_statefulpartitionedcall_args_8$vgg19_statefulpartitionedcall_args_9%vgg19_statefulpartitionedcall_args_10%vgg19_statefulpartitionedcall_args_11%vgg19_statefulpartitionedcall_args_12%vgg19_statefulpartitionedcall_args_13%vgg19_statefulpartitionedcall_args_14%vgg19_statefulpartitionedcall_args_15%vgg19_statefulpartitionedcall_args_16%vgg19_statefulpartitionedcall_args_17%vgg19_statefulpartitionedcall_args_18%vgg19_statefulpartitionedcall_args_19%vgg19_statefulpartitionedcall_args_20%vgg19_statefulpartitionedcall_args_21%vgg19_statefulpartitionedcall_args_22%vgg19_statefulpartitionedcall_args_23%vgg19_statefulpartitionedcall_args_24%vgg19_statefulpartitionedcall_args_25%vgg19_statefulpartitionedcall_args_26%vgg19_statefulpartitionedcall_args_27%vgg19_statefulpartitionedcall_args_28%vgg19_statefulpartitionedcall_args_29%vgg19_statefulpartitionedcall_args_30%vgg19_statefulpartitionedcall_args_31%vgg19_statefulpartitionedcall_args_32*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274257*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*,
Tin%
#2!*-
_gradient_op_typePartitionedCall-274298Ќ
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*)
_output_shapes
:€€€€€€€€€Аƒ*-
_gradient_op_typePartitionedCall-274379*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_274373«
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*)
_output_shapes
:€€€€€€€€€Аƒ*-
_gradient_op_typePartitionedCall-274425*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274413*
Tout
2v
	tags/CastCast dropout/PartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒИ
tags/StatefulPartitionedCallStatefulPartitionedCalltags/Cast:y:0#tags_statefulpartitionedcall_args_1#tags_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-274448*I
fDRB
@__inference_tags_layer_call_and_return_conditional_losses_274442*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*'
_output_shapes
:€€€€€€€€€u
cc3/CastCast dropout/PartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒГ
cc3/StatefulPartitionedCallStatefulPartitionedCallcc3/Cast:y:0"cc3_statefulpartitionedcall_args_1"cc3_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-274477*H
fCRA
?__inference_cc3_layer_call_and_return_conditional_losses_274471*
Tout
2…
IdentityIdentity$cc3/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0ћ

Identity_1Identity%tags/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall2<
tags/StatefulPartitionedCalltags/StatefulPartitionedCall2:
cc3/StatefulPartitionedCallcc3/StatefulPartitionedCall:% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ 
и
Щ
&__inference_model_layer_call_fn_274633	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36
identity

identity_1ИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36*0
Tin)
'2%*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*-
_gradient_op_typePartitionedCall-274592*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_274591*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄВ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0Д

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:€€€€€€€€€*
T0"!

identity_1Identity_1:output:0"
identityIdentity:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :  :! :" :# :$ :% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
Н
б
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЂ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@Ађ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
л
ц	
&__inference_vgg19_layer_call_fn_275483

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallс

StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_273974*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273975Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
«
a
(__inference_dropout_layer_call_fn_275842

inputs
identityИҐStatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputs*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274406*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*)
_output_shapes
:€€€€€€€€€Аƒ*-
_gradient_op_typePartitionedCall-274417Д
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"
identityIdentity:output:0*(
_input_shapes
:€€€€€€€€€Аƒ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
№
¶
%__inference_tags_layer_call_fn_275883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-274448*I
fDRB
@__inference_tags_layer_call_and_return_conditional_losses_274442*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
∆
Ч
$__inference_signature_wrapper_274780	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36
identity

identity_1ИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36**
f%R#
!__inference__wrapped_model_273213*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
Tin)
'2%*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*-
_gradient_op_typePartitionedCall-274739В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€Д

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :  :! :" :# :$ :% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : 
Ѓ
b
C__inference_dropout_layer_call_and_return_conditional_losses_274406

inputs
identityИO
dropout/rateConst*
value
B jАh*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:\
dropout/random_uniform/minConst*
value	B j *
dtype0*
_output_shapes
: ]
dropout/random_uniform/maxConst*
value
B jАx*
dtype0*
_output_shapes
: О
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: §
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0Ц
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒP
dropout/sub/xConst*
dtype0*
_output_shapes
: *
value
B jАxb
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: T
dropout/truediv/xConst*
value
B jАx*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Л
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒc
dropout/mulMulinputsdropout/truediv:z:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*)
_output_shapes
:€€€€€€€€€Аƒ*

SrcT0
k
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ[
IdentityIdentitydropout/mul_1:z:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0"
identityIdentity:output:0*(
_input_shapes
:€€€€€€€€€Аƒ:& "
 
_user_specified_nameinputs
л
ц	
&__inference_vgg19_layer_call_fn_275764

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallс

StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-274260*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274135*
Tout
2Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : : : : : : :  :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
П
б
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ї
Ѓ
-__inference_block3_conv4_layer_call_fn_273447

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273442*Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ѕ	
ў
@__inference_tags_layer_call_and_return_conditional_losses_275876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
э
_
C__inference_flatten_layer_call_and_return_conditional_losses_275807

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€ b  f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
≤≠
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_274257

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpl
block1_conv1/CastCastinputs*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€ааƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ƒ
block1_conv1/Conv2DConv2Dblock1_conv1/Cast:y:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@Ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@Ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€аа@*
T0µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:€€€€€€€€€pp@≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
*
paddingSAMEї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAMEї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€АЭ

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
П
б
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
*
paddingSAME°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
лl
р
A__inference_vgg19_layer_call_and_return_conditional_losses_273877

inputs/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block3_conv4_statefulpartitionedcall_args_1/
+block3_conv4_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block4_conv4_statefulpartitionedcall_args_1/
+block4_conv4_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2/
+block5_conv4_statefulpartitionedcall_args_1/
+block5_conv4_statefulpartitionedcall_args_2
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCallЂ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273233*Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*1
_output_shapes
:€€€€€€€€€аа@*
Tin
2“
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*1
_output_shapes
:€€€€€€€€€аа@*-
_gradient_op_typePartitionedCall-273258*Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252*
Tout
2в
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*/
_output_shapes
:€€€€€€€€€pp@*-
_gradient_op_typePartitionedCall-273277*P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271*
Tout
2»
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€ppА*-
_gradient_op_typePartitionedCall-273300—
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€ppА*-
_gradient_op_typePartitionedCall-273325*Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319*
Tout
2г
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273344*P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А»
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€88А*
Tin
2*-
_gradient_op_typePartitionedCall-273367*Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361—
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*0
_output_shapes
:€€€€€€€€€88А*
Tin
2*-
_gradient_op_typePartitionedCall-273392*Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273417*Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411—
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0+block3_conv4_statefulpartitionedcall_args_1+block3_conv4_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273442*Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436*
Tout
2г
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273461*P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2»
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273484*Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2—
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273509*Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503*
Tout
2—
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273534*Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0+block4_conv4_statefulpartitionedcall_args_1+block4_conv4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273559*Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€Аг
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273578»
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273601*Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273626*Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2—
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273651*Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645—
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0+block5_conv4_statefulpartitionedcall_args_1+block5_conv4_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273676*Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄг
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273695*P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2е
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:  :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : 
ѕ	
ў
@__inference_tags_layer_call_and_return_conditional_losses_274442

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:€€€€€€€€€*
T0†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:€€€€€€€€€*
T0Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ї
Ѓ
-__inference_block5_conv3_layer_call_fn_273656

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273651*Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645*
Tout
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ї
Ѓ
-__inference_block3_conv2_layer_call_fn_273397

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273392*Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
≤≠
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_274135

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpl
block1_conv1/CastCastinputs*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€ааƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ƒ
block1_conv1/Conv2DConv2Dblock1_conv1/Cast:y:0*block1_conv1/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
*
paddingSAMEЇ
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€аа@*
T0ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
*
paddingSAMEЇ
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppАї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0ґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAMEї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
ї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0ґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALID∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALIDЭ

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
П
б
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
*
paddingSAME°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Є
Ѓ
-__inference_block1_conv2_layer_call_fn_273263

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273258*Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
В¬
я
"__inference__traced_restore_276240
file_prefix!
assignvariableop_cc3_2_kernel!
assignvariableop_1_cc3_2_bias$
 assignvariableop_2_tags_2_kernel"
assignvariableop_3_tags_2_bias+
'assignvariableop_4_training_1_adam_iter-
)assignvariableop_5_training_1_adam_beta_1-
)assignvariableop_6_training_1_adam_beta_2,
(assignvariableop_7_training_1_adam_decay4
0assignvariableop_8_training_1_adam_learning_rate,
(assignvariableop_9_block1_conv1_2_kernel+
'assignvariableop_10_block1_conv1_2_bias-
)assignvariableop_11_block1_conv2_2_kernel+
'assignvariableop_12_block1_conv2_2_bias-
)assignvariableop_13_block2_conv1_2_kernel+
'assignvariableop_14_block2_conv1_2_bias-
)assignvariableop_15_block2_conv2_2_kernel+
'assignvariableop_16_block2_conv2_2_bias-
)assignvariableop_17_block3_conv1_2_kernel+
'assignvariableop_18_block3_conv1_2_bias-
)assignvariableop_19_block3_conv2_2_kernel+
'assignvariableop_20_block3_conv2_2_bias-
)assignvariableop_21_block3_conv3_2_kernel+
'assignvariableop_22_block3_conv3_2_bias-
)assignvariableop_23_block3_conv4_2_kernel+
'assignvariableop_24_block3_conv4_2_bias-
)assignvariableop_25_block4_conv1_2_kernel+
'assignvariableop_26_block4_conv1_2_bias-
)assignvariableop_27_block4_conv2_2_kernel+
'assignvariableop_28_block4_conv2_2_bias-
)assignvariableop_29_block4_conv3_2_kernel+
'assignvariableop_30_block4_conv3_2_bias-
)assignvariableop_31_block4_conv4_2_kernel+
'assignvariableop_32_block4_conv4_2_bias-
)assignvariableop_33_block5_conv1_2_kernel+
'assignvariableop_34_block5_conv1_2_bias-
)assignvariableop_35_block5_conv2_2_kernel+
'assignvariableop_36_block5_conv2_2_bias-
)assignvariableop_37_block5_conv3_2_kernel+
'assignvariableop_38_block5_conv3_2_bias-
)assignvariableop_39_block5_conv4_2_kernel+
'assignvariableop_40_block5_conv4_2_bias
assignvariableop_41_total
assignvariableop_42_count
assignvariableop_43_total_1
assignvariableop_44_count_16
2assignvariableop_45_training_1_adam_cc3_2_kernel_m4
0assignvariableop_46_training_1_adam_cc3_2_bias_m7
3assignvariableop_47_training_1_adam_tags_2_kernel_m5
1assignvariableop_48_training_1_adam_tags_2_bias_m6
2assignvariableop_49_training_1_adam_cc3_2_kernel_v4
0assignvariableop_50_training_1_adam_cc3_2_bias_v7
3assignvariableop_51_training_1_adam_tags_2_kernel_v5
1assignvariableop_52_training_1_adam_tags_2_bias_v
identity_54ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1ґ
RestoreV2/tensor_namesConst"/device:CPU:0*№
value“Bѕ5B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:5Џ
RestoreV2/shape_and_slicesConst"/device:CPU:0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:5™
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*C
dtypes9
725	*к
_output_shapes„
‘:::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0y
AssignVariableOpAssignVariableOpassignvariableop_cc3_2_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0}
AssignVariableOp_1AssignVariableOpassignvariableop_1_cc3_2_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:А
AssignVariableOp_2AssignVariableOp assignvariableop_2_tags_2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:~
AssignVariableOp_3AssignVariableOpassignvariableop_3_tags_2_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:З
AssignVariableOp_4AssignVariableOp'assignvariableop_4_training_1_adam_iterIdentity_4:output:0*
dtype0	*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:Й
AssignVariableOp_5AssignVariableOp)assignvariableop_5_training_1_adam_beta_1Identity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp)assignvariableop_6_training_1_adam_beta_2Identity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:И
AssignVariableOp_7AssignVariableOp(assignvariableop_7_training_1_adam_decayIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0Р
AssignVariableOp_8AssignVariableOp0assignvariableop_8_training_1_adam_learning_rateIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:И
AssignVariableOp_9AssignVariableOp(assignvariableop_9_block1_conv1_2_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Й
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block1_conv1_2_biasIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Л
AssignVariableOp_11AssignVariableOp)assignvariableop_11_block1_conv2_2_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Й
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block1_conv2_2_biasIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:Л
AssignVariableOp_13AssignVariableOp)assignvariableop_13_block2_conv1_2_kernelIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Й
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block2_conv1_2_biasIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Л
AssignVariableOp_15AssignVariableOp)assignvariableop_15_block2_conv2_2_kernelIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Й
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block2_conv2_2_biasIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Л
AssignVariableOp_17AssignVariableOp)assignvariableop_17_block3_conv1_2_kernelIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Й
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block3_conv1_2_biasIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:Л
AssignVariableOp_19AssignVariableOp)assignvariableop_19_block3_conv2_2_kernelIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Й
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block3_conv2_2_biasIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0Л
AssignVariableOp_21AssignVariableOp)assignvariableop_21_block3_conv3_2_kernelIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:Й
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block3_conv3_2_biasIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:Л
AssignVariableOp_23AssignVariableOp)assignvariableop_23_block3_conv4_2_kernelIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Й
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block3_conv4_2_biasIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOp)assignvariableop_25_block4_conv1_2_kernelIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:Й
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block4_conv1_2_biasIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:Л
AssignVariableOp_27AssignVariableOp)assignvariableop_27_block4_conv2_2_kernelIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0Й
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block4_conv2_2_biasIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:Л
AssignVariableOp_29AssignVariableOp)assignvariableop_29_block4_conv3_2_kernelIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:Й
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block4_conv3_2_biasIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:Л
AssignVariableOp_31AssignVariableOp)assignvariableop_31_block4_conv4_2_kernelIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:Й
AssignVariableOp_32AssignVariableOp'assignvariableop_32_block4_conv4_2_biasIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0Л
AssignVariableOp_33AssignVariableOp)assignvariableop_33_block5_conv1_2_kernelIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:Й
AssignVariableOp_34AssignVariableOp'assignvariableop_34_block5_conv1_2_biasIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:Л
AssignVariableOp_35AssignVariableOp)assignvariableop_35_block5_conv2_2_kernelIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:Й
AssignVariableOp_36AssignVariableOp'assignvariableop_36_block5_conv2_2_biasIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:Л
AssignVariableOp_37AssignVariableOp)assignvariableop_37_block5_conv3_2_kernelIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:Й
AssignVariableOp_38AssignVariableOp'assignvariableop_38_block5_conv3_2_biasIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:Л
AssignVariableOp_39AssignVariableOp)assignvariableop_39_block5_conv4_2_kernelIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:Й
AssignVariableOp_40AssignVariableOp'assignvariableop_40_block5_conv4_2_biasIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:{
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:{
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:}
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:}
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:Ф
AssignVariableOp_45AssignVariableOp2assignvariableop_45_training_1_adam_cc3_2_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0Т
AssignVariableOp_46AssignVariableOp0assignvariableop_46_training_1_adam_cc3_2_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:Х
AssignVariableOp_47AssignVariableOp3assignvariableop_47_training_1_adam_tags_2_kernel_mIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0У
AssignVariableOp_48AssignVariableOp1assignvariableop_48_training_1_adam_tags_2_bias_mIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:Ф
AssignVariableOp_49AssignVariableOp2assignvariableop_49_training_1_adam_cc3_2_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0Т
AssignVariableOp_50AssignVariableOp0assignvariableop_50_training_1_adam_cc3_2_bias_vIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0Х
AssignVariableOp_51AssignVariableOp3assignvariableop_51_training_1_adam_tags_2_kernel_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0У
AssignVariableOp_52AssignVariableOp1assignvariableop_52_training_1_adam_tags_2_bias_vIdentity_52:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ё	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0к	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_54Identity_54:output:0*л
_input_shapesў
÷: :::::::::::::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492
RestoreV2_1RestoreV2_1:" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
Ґ
c
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689

inputs
identityЂ
MaxPoolMaxPoolinputs*
T0*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
о
ч	
&__inference_vgg19_layer_call_fn_273913
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallт

StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_273877*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273878Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
о
ч	
&__inference_vgg19_layer_call_fn_274010
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallт

StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*-
_gradient_op_typePartitionedCall-273975*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_273974*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€АЛ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :  :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : 
ї
Ѓ
-__inference_block5_conv2_layer_call_fn_273631

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273626*Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
µђ
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_275409

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0Ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@Ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
*
ksize
*
paddingVALID≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppАї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
*
paddingSAMEї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0ґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
ksize
*
paddingVALID∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0ґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0ґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALIDЭ

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
‘	
Ў
?__inference_cc3_layer_call_and_return_conditional_losses_274471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
T0"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ј1
з
A__inference_model_layer_call_and_return_conditional_losses_274490	
input(
$vgg19_statefulpartitionedcall_args_1(
$vgg19_statefulpartitionedcall_args_2(
$vgg19_statefulpartitionedcall_args_3(
$vgg19_statefulpartitionedcall_args_4(
$vgg19_statefulpartitionedcall_args_5(
$vgg19_statefulpartitionedcall_args_6(
$vgg19_statefulpartitionedcall_args_7(
$vgg19_statefulpartitionedcall_args_8(
$vgg19_statefulpartitionedcall_args_9)
%vgg19_statefulpartitionedcall_args_10)
%vgg19_statefulpartitionedcall_args_11)
%vgg19_statefulpartitionedcall_args_12)
%vgg19_statefulpartitionedcall_args_13)
%vgg19_statefulpartitionedcall_args_14)
%vgg19_statefulpartitionedcall_args_15)
%vgg19_statefulpartitionedcall_args_16)
%vgg19_statefulpartitionedcall_args_17)
%vgg19_statefulpartitionedcall_args_18)
%vgg19_statefulpartitionedcall_args_19)
%vgg19_statefulpartitionedcall_args_20)
%vgg19_statefulpartitionedcall_args_21)
%vgg19_statefulpartitionedcall_args_22)
%vgg19_statefulpartitionedcall_args_23)
%vgg19_statefulpartitionedcall_args_24)
%vgg19_statefulpartitionedcall_args_25)
%vgg19_statefulpartitionedcall_args_26)
%vgg19_statefulpartitionedcall_args_27)
%vgg19_statefulpartitionedcall_args_28)
%vgg19_statefulpartitionedcall_args_29)
%vgg19_statefulpartitionedcall_args_30)
%vgg19_statefulpartitionedcall_args_31)
%vgg19_statefulpartitionedcall_args_32'
#tags_statefulpartitionedcall_args_1'
#tags_statefulpartitionedcall_args_2&
"cc3_statefulpartitionedcall_args_1&
"cc3_statefulpartitionedcall_args_2
identity

identity_1ИҐcc3/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐtags/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallґ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput$vgg19_statefulpartitionedcall_args_1$vgg19_statefulpartitionedcall_args_2$vgg19_statefulpartitionedcall_args_3$vgg19_statefulpartitionedcall_args_4$vgg19_statefulpartitionedcall_args_5$vgg19_statefulpartitionedcall_args_6$vgg19_statefulpartitionedcall_args_7$vgg19_statefulpartitionedcall_args_8$vgg19_statefulpartitionedcall_args_9%vgg19_statefulpartitionedcall_args_10%vgg19_statefulpartitionedcall_args_11%vgg19_statefulpartitionedcall_args_12%vgg19_statefulpartitionedcall_args_13%vgg19_statefulpartitionedcall_args_14%vgg19_statefulpartitionedcall_args_15%vgg19_statefulpartitionedcall_args_16%vgg19_statefulpartitionedcall_args_17%vgg19_statefulpartitionedcall_args_18%vgg19_statefulpartitionedcall_args_19%vgg19_statefulpartitionedcall_args_20%vgg19_statefulpartitionedcall_args_21%vgg19_statefulpartitionedcall_args_22%vgg19_statefulpartitionedcall_args_23%vgg19_statefulpartitionedcall_args_24%vgg19_statefulpartitionedcall_args_25%vgg19_statefulpartitionedcall_args_26%vgg19_statefulpartitionedcall_args_27%vgg19_statefulpartitionedcall_args_28%vgg19_statefulpartitionedcall_args_29%vgg19_statefulpartitionedcall_args_30%vgg19_statefulpartitionedcall_args_31%vgg19_statefulpartitionedcall_args_32*-
_gradient_op_typePartitionedCall-274260*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274135*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€АЌ
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-274379*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_274373*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2„
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274406*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2*-
_gradient_op_typePartitionedCall-274417~
	tags/CastCast(dropout/StatefulPartitionedCall:output:0*

DstT0*)
_output_shapes
:€€€€€€€€€Аƒ*

SrcT0И
tags/StatefulPartitionedCallStatefulPartitionedCalltags/Cast:y:0#tags_statefulpartitionedcall_args_1#tags_statefulpartitionedcall_args_2*I
fDRB
@__inference_tags_layer_call_and_return_conditional_losses_274442*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*'
_output_shapes
:€€€€€€€€€*-
_gradient_op_typePartitionedCall-274448}
cc3/CastCast(dropout/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒГ
cc3/StatefulPartitionedCallStatefulPartitionedCallcc3/Cast:y:0"cc3_statefulpartitionedcall_args_1"cc3_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-274477*H
fCRA
?__inference_cc3_layer_call_and_return_conditional_losses_274471л
IdentityIdentity$cc3/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€о

Identity_1Identity%tags/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0"
identityIdentity:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2<
tags/StatefulPartitionedCalltags/StatefulPartitionedCall2:
cc3/StatefulPartitionedCallcc3/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ 
ї
Ѓ
-__inference_block5_conv1_layer_call_fn_273606

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273601*Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
П
б
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Џ
•
$__inference_cc3_layer_call_fn_275865

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-274477*H
fCRA
?__inference_cc3_layer_call_and_return_conditional_losses_274471*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*'
_output_shapes
:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Є
Ѓ
-__inference_block1_conv1_layer_call_fn_273238

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*-
_gradient_op_typePartitionedCall-273233*Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227*
Tout
2Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
П
б
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
*
paddingSAME°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0k
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
В[
ы
__inference__traced_save_276068
file_prefix+
'savev2_cc3_2_kernel_read_readvariableop)
%savev2_cc3_2_bias_read_readvariableop,
(savev2_tags_2_kernel_read_readvariableop*
&savev2_tags_2_bias_read_readvariableop3
/savev2_training_1_adam_iter_read_readvariableop	5
1savev2_training_1_adam_beta_1_read_readvariableop5
1savev2_training_1_adam_beta_2_read_readvariableop4
0savev2_training_1_adam_decay_read_readvariableop<
8savev2_training_1_adam_learning_rate_read_readvariableop4
0savev2_block1_conv1_2_kernel_read_readvariableop2
.savev2_block1_conv1_2_bias_read_readvariableop4
0savev2_block1_conv2_2_kernel_read_readvariableop2
.savev2_block1_conv2_2_bias_read_readvariableop4
0savev2_block2_conv1_2_kernel_read_readvariableop2
.savev2_block2_conv1_2_bias_read_readvariableop4
0savev2_block2_conv2_2_kernel_read_readvariableop2
.savev2_block2_conv2_2_bias_read_readvariableop4
0savev2_block3_conv1_2_kernel_read_readvariableop2
.savev2_block3_conv1_2_bias_read_readvariableop4
0savev2_block3_conv2_2_kernel_read_readvariableop2
.savev2_block3_conv2_2_bias_read_readvariableop4
0savev2_block3_conv3_2_kernel_read_readvariableop2
.savev2_block3_conv3_2_bias_read_readvariableop4
0savev2_block3_conv4_2_kernel_read_readvariableop2
.savev2_block3_conv4_2_bias_read_readvariableop4
0savev2_block4_conv1_2_kernel_read_readvariableop2
.savev2_block4_conv1_2_bias_read_readvariableop4
0savev2_block4_conv2_2_kernel_read_readvariableop2
.savev2_block4_conv2_2_bias_read_readvariableop4
0savev2_block4_conv3_2_kernel_read_readvariableop2
.savev2_block4_conv3_2_bias_read_readvariableop4
0savev2_block4_conv4_2_kernel_read_readvariableop2
.savev2_block4_conv4_2_bias_read_readvariableop4
0savev2_block5_conv1_2_kernel_read_readvariableop2
.savev2_block5_conv1_2_bias_read_readvariableop4
0savev2_block5_conv2_2_kernel_read_readvariableop2
.savev2_block5_conv2_2_bias_read_readvariableop4
0savev2_block5_conv3_2_kernel_read_readvariableop2
.savev2_block5_conv3_2_bias_read_readvariableop4
0savev2_block5_conv4_2_kernel_read_readvariableop2
.savev2_block5_conv4_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop=
9savev2_training_1_adam_cc3_2_kernel_m_read_readvariableop;
7savev2_training_1_adam_cc3_2_bias_m_read_readvariableop>
:savev2_training_1_adam_tags_2_kernel_m_read_readvariableop<
8savev2_training_1_adam_tags_2_bias_m_read_readvariableop=
9savev2_training_1_adam_cc3_2_kernel_v_read_readvariableop;
7savev2_training_1_adam_cc3_2_bias_v_read_readvariableop>
:savev2_training_1_adam_tags_2_kernel_v_read_readvariableop<
8savev2_training_1_adam_tags_2_bias_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_676e8a826cb84bf2b1a7d9c93cf4fe3c/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ≥
SaveV2/tensor_namesConst"/device:CPU:0*№
value“Bѕ5B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:5„
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:5*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_cc3_2_kernel_read_readvariableop%savev2_cc3_2_bias_read_readvariableop(savev2_tags_2_kernel_read_readvariableop&savev2_tags_2_bias_read_readvariableop/savev2_training_1_adam_iter_read_readvariableop1savev2_training_1_adam_beta_1_read_readvariableop1savev2_training_1_adam_beta_2_read_readvariableop0savev2_training_1_adam_decay_read_readvariableop8savev2_training_1_adam_learning_rate_read_readvariableop0savev2_block1_conv1_2_kernel_read_readvariableop.savev2_block1_conv1_2_bias_read_readvariableop0savev2_block1_conv2_2_kernel_read_readvariableop.savev2_block1_conv2_2_bias_read_readvariableop0savev2_block2_conv1_2_kernel_read_readvariableop.savev2_block2_conv1_2_bias_read_readvariableop0savev2_block2_conv2_2_kernel_read_readvariableop.savev2_block2_conv2_2_bias_read_readvariableop0savev2_block3_conv1_2_kernel_read_readvariableop.savev2_block3_conv1_2_bias_read_readvariableop0savev2_block3_conv2_2_kernel_read_readvariableop.savev2_block3_conv2_2_bias_read_readvariableop0savev2_block3_conv3_2_kernel_read_readvariableop.savev2_block3_conv3_2_bias_read_readvariableop0savev2_block3_conv4_2_kernel_read_readvariableop.savev2_block3_conv4_2_bias_read_readvariableop0savev2_block4_conv1_2_kernel_read_readvariableop.savev2_block4_conv1_2_bias_read_readvariableop0savev2_block4_conv2_2_kernel_read_readvariableop.savev2_block4_conv2_2_bias_read_readvariableop0savev2_block4_conv3_2_kernel_read_readvariableop.savev2_block4_conv3_2_bias_read_readvariableop0savev2_block4_conv4_2_kernel_read_readvariableop.savev2_block4_conv4_2_bias_read_readvariableop0savev2_block5_conv1_2_kernel_read_readvariableop.savev2_block5_conv1_2_bias_read_readvariableop0savev2_block5_conv2_2_kernel_read_readvariableop.savev2_block5_conv2_2_bias_read_readvariableop0savev2_block5_conv3_2_kernel_read_readvariableop.savev2_block5_conv3_2_bias_read_readvariableop0savev2_block5_conv4_2_kernel_read_readvariableop.savev2_block5_conv4_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop9savev2_training_1_adam_cc3_2_kernel_m_read_readvariableop7savev2_training_1_adam_cc3_2_bias_m_read_readvariableop:savev2_training_1_adam_tags_2_kernel_m_read_readvariableop8savev2_training_1_adam_tags_2_bias_m_read_readvariableop9savev2_training_1_adam_cc3_2_kernel_v_read_readvariableop7savev2_training_1_adam_cc3_2_bias_v_read_readvariableop:savev2_training_1_adam_tags_2_kernel_v_read_readvariableop8savev2_training_1_adam_tags_2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:√
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2є
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*ј
_input_shapesЃ
Ђ: :
Аƒ::
Аƒ:: : : : : :@:@:@@:@:@А:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А: : : : :
Аƒ::
Аƒ::
Аƒ::
Аƒ:: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:4 :5 :6 :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 
ья
и
A__inference_model_layer_call_and_return_conditional_losses_275081

inputs5
1vgg19_block1_conv1_conv2d_readvariableop_resource6
2vgg19_block1_conv1_biasadd_readvariableop_resource5
1vgg19_block1_conv2_conv2d_readvariableop_resource6
2vgg19_block1_conv2_biasadd_readvariableop_resource5
1vgg19_block2_conv1_conv2d_readvariableop_resource6
2vgg19_block2_conv1_biasadd_readvariableop_resource5
1vgg19_block2_conv2_conv2d_readvariableop_resource6
2vgg19_block2_conv2_biasadd_readvariableop_resource5
1vgg19_block3_conv1_conv2d_readvariableop_resource6
2vgg19_block3_conv1_biasadd_readvariableop_resource5
1vgg19_block3_conv2_conv2d_readvariableop_resource6
2vgg19_block3_conv2_biasadd_readvariableop_resource5
1vgg19_block3_conv3_conv2d_readvariableop_resource6
2vgg19_block3_conv3_biasadd_readvariableop_resource5
1vgg19_block3_conv4_conv2d_readvariableop_resource6
2vgg19_block3_conv4_biasadd_readvariableop_resource5
1vgg19_block4_conv1_conv2d_readvariableop_resource6
2vgg19_block4_conv1_biasadd_readvariableop_resource5
1vgg19_block4_conv2_conv2d_readvariableop_resource6
2vgg19_block4_conv2_biasadd_readvariableop_resource5
1vgg19_block4_conv3_conv2d_readvariableop_resource6
2vgg19_block4_conv3_biasadd_readvariableop_resource5
1vgg19_block4_conv4_conv2d_readvariableop_resource6
2vgg19_block4_conv4_biasadd_readvariableop_resource5
1vgg19_block5_conv1_conv2d_readvariableop_resource6
2vgg19_block5_conv1_biasadd_readvariableop_resource5
1vgg19_block5_conv2_conv2d_readvariableop_resource6
2vgg19_block5_conv2_biasadd_readvariableop_resource5
1vgg19_block5_conv3_conv2d_readvariableop_resource6
2vgg19_block5_conv3_biasadd_readvariableop_resource5
1vgg19_block5_conv4_conv2d_readvariableop_resource6
2vgg19_block5_conv4_biasadd_readvariableop_resource'
#tags_matmul_readvariableop_resource(
$tags_biasadd_readvariableop_resource&
"cc3_matmul_readvariableop_resource'
#cc3_biasadd_readvariableop_resource
identity

identity_1ИҐcc3/BiasAdd/ReadVariableOpҐcc3/MatMul/ReadVariableOpҐtags/BiasAdd/ReadVariableOpҐtags/MatMul/ReadVariableOpҐ)vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv1/Conv2D/ReadVariableOpҐ)vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv2/Conv2D/ReadVariableOpҐ)vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv1/Conv2D/ReadVariableOpҐ)vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv1/Conv2D/ReadVariableOpҐ)vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv3/Conv2D/ReadVariableOpҐ)vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv4/Conv2D/ReadVariableOpҐ)vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv1/Conv2D/ReadVariableOpҐ)vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv2/Conv2D/ReadVariableOpҐ)vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv3/Conv2D/ReadVariableOpҐ)vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv4/Conv2D/ReadVariableOpҐ)vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv1/Conv2D/ReadVariableOpҐ)vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv2/Conv2D/ReadVariableOpҐ)vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv3/Conv2D/ReadVariableOpҐ)vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv4/Conv2D/ReadVariableOpr
vgg19/block1_conv1/CastCastinputs*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€аа–
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@÷
vgg19/block1_conv1/Conv2DConv2Dvgg19/block1_conv1/Cast:y:00vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
∆
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Є
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@А
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@–
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@а
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@∆
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Є
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0А
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@Ѕ
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
*
ksize
*
paddingVALID—
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А№
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0«
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА“
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0«
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА¬
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А*
T0“
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
«
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А“
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
*
paddingSAME«
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А“
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
«
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А“
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
«
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А¬
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0“
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAME«
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А¬
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА№
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0«
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А«
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А“
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААя
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
«
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АЈ
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А¬
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
f
flatten/Reshape/shapeConst*
valueB"€€€€ b  *
dtype0*
_output_shapes
:Т
flatten/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten/Reshape/shape:output:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0j
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒo
	tags/CastCastdropout/Identity:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒЃ
tags/MatMul/ReadVariableOpReadVariableOp#tags_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒz
tags/MatMulMatMultags/Cast:y:0"tags/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
tags/BiasAdd/ReadVariableOpReadVariableOp$tags_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Е
tags/BiasAddBiasAddtags/MatMul:product:0#tags/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
tags/SigmoidSigmoidtags/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
cc3/CastCastdropout/Identity:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€Аƒђ
cc3/MatMul/ReadVariableOpReadVariableOp"cc3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒw

cc3/MatMulMatMulcc3/Cast:y:0!cc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€®
cc3/BiasAdd/ReadVariableOpReadVariableOp#cc3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:В
cc3/BiasAddBiasAddcc3/MatMul:product:0"cc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
cc3/SoftmaxSoftmaxcc3/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€*
T0Ѕ
IdentityIdentitycc3/Softmax:softmax:0^cc3/BiasAdd/ReadVariableOp^cc3/MatMul/ReadVariableOp^tags/BiasAdd/ReadVariableOp^tags/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
T0Њ

Identity_1Identitytags/Sigmoid:y:0^cc3/BiasAdd/ReadVariableOp^cc3/MatMul/ReadVariableOp^tags/BiasAdd/ReadVariableOp^tags/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp28
tags/MatMul/ReadVariableOptags/MatMul/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp28
cc3/BiasAdd/ReadVariableOpcc3/BiasAdd/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp26
cc3/MatMul/ReadVariableOpcc3/MatMul/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2:
tags/BiasAdd/ReadVariableOptags/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp: : : : : : : : : : : : : : :  :! :" :# :$ :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : 
ї
Ѓ
-__inference_block4_conv2_layer_call_fn_273514

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273509*Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
≤≠
ј
A__inference_vgg19_layer_call_and_return_conditional_losses_275605

inputs/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block3_conv4_conv2d_readvariableop_resource0
,block3_conv4_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block4_conv4_conv2d_readvariableop_resource0
,block4_conv4_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource/
+block5_conv4_conv2d_readvariableop_resource0
,block5_conv4_biasadd_readvariableop_resource
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpl
block1_conv1/CastCastinputs*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€ааƒ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ƒ
block1_conv1/Conv2DConv2Dblock1_conv1/Cast:y:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
Ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@ƒ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@ќ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
Ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@µ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:€€€€€€€€€pp@*
T0≈
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@А 
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€ppА*
T0∆
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppАї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАґ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88А*
T0∆
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0ї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€88А*
T0s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А∆
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88Аї
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88Аs
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€88А*
T0ґ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALID∆
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аґ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0∆
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:АА 
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€Аї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAMEї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А∆
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААЌ
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0ї
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А•
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0ґ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0Э

IdentityIdentityblock5_pool/MaxPool:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
Ѓ
b
C__inference_dropout_layer_call_and_return_conditional_losses_275832

inputs
identityИO
dropout/rateConst*
value
B jАh*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:\
dropout/random_uniform/minConst*
value	B j *
dtype0*
_output_shapes
: ]
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
value
B jАxО
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*)
_output_shapes
:€€€€€€€€€АƒМ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0§
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*)
_output_shapes
:€€€€€€€€€АƒЦ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*)
_output_shapes
:€€€€€€€€€АƒP
dropout/sub/xConst*
dtype0*
_output_shapes
: *
value
B jАxb
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0T
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
value
B jАxh
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Л
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒc
dropout/mulMulinputsdropout/truediv:z:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0q
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*)
_output_shapes
:€€€€€€€€€Аƒ*

SrcT0
k
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*)
_output_shapes
:€€€€€€€€€Аƒ*
T0[
IdentityIdentitydropout/mul_1:z:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"
identityIdentity:output:0*(
_input_shapes
:€€€€€€€€€Аƒ:& "
 
_user_specified_nameinputs
ї
Ѓ
-__inference_block5_conv4_layer_call_fn_273681

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273676*Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
л
Ъ
&__inference_model_layer_call_fn_275124

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36
identity

identity_1ИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36*-
_gradient_op_typePartitionedCall-274592*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_274591*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
Tin)
'2%*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€Д

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0"
identityIdentity:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :& "
 
_user_specified_nameinputs: : : : : 
‘	
Ў
?__inference_cc3_layer_call_and_return_conditional_losses_275858

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOp§
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Аƒi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€Аƒ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
З
б
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp™
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@•
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
щ/
∆
A__inference_model_layer_call_and_return_conditional_losses_274685

inputs(
$vgg19_statefulpartitionedcall_args_1(
$vgg19_statefulpartitionedcall_args_2(
$vgg19_statefulpartitionedcall_args_3(
$vgg19_statefulpartitionedcall_args_4(
$vgg19_statefulpartitionedcall_args_5(
$vgg19_statefulpartitionedcall_args_6(
$vgg19_statefulpartitionedcall_args_7(
$vgg19_statefulpartitionedcall_args_8(
$vgg19_statefulpartitionedcall_args_9)
%vgg19_statefulpartitionedcall_args_10)
%vgg19_statefulpartitionedcall_args_11)
%vgg19_statefulpartitionedcall_args_12)
%vgg19_statefulpartitionedcall_args_13)
%vgg19_statefulpartitionedcall_args_14)
%vgg19_statefulpartitionedcall_args_15)
%vgg19_statefulpartitionedcall_args_16)
%vgg19_statefulpartitionedcall_args_17)
%vgg19_statefulpartitionedcall_args_18)
%vgg19_statefulpartitionedcall_args_19)
%vgg19_statefulpartitionedcall_args_20)
%vgg19_statefulpartitionedcall_args_21)
%vgg19_statefulpartitionedcall_args_22)
%vgg19_statefulpartitionedcall_args_23)
%vgg19_statefulpartitionedcall_args_24)
%vgg19_statefulpartitionedcall_args_25)
%vgg19_statefulpartitionedcall_args_26)
%vgg19_statefulpartitionedcall_args_27)
%vgg19_statefulpartitionedcall_args_28)
%vgg19_statefulpartitionedcall_args_29)
%vgg19_statefulpartitionedcall_args_30)
%vgg19_statefulpartitionedcall_args_31)
%vgg19_statefulpartitionedcall_args_32'
#tags_statefulpartitionedcall_args_1'
#tags_statefulpartitionedcall_args_2&
"cc3_statefulpartitionedcall_args_1&
"cc3_statefulpartitionedcall_args_2
identity

identity_1ИҐcc3/StatefulPartitionedCallҐtags/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallЈ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputs$vgg19_statefulpartitionedcall_args_1$vgg19_statefulpartitionedcall_args_2$vgg19_statefulpartitionedcall_args_3$vgg19_statefulpartitionedcall_args_4$vgg19_statefulpartitionedcall_args_5$vgg19_statefulpartitionedcall_args_6$vgg19_statefulpartitionedcall_args_7$vgg19_statefulpartitionedcall_args_8$vgg19_statefulpartitionedcall_args_9%vgg19_statefulpartitionedcall_args_10%vgg19_statefulpartitionedcall_args_11%vgg19_statefulpartitionedcall_args_12%vgg19_statefulpartitionedcall_args_13%vgg19_statefulpartitionedcall_args_14%vgg19_statefulpartitionedcall_args_15%vgg19_statefulpartitionedcall_args_16%vgg19_statefulpartitionedcall_args_17%vgg19_statefulpartitionedcall_args_18%vgg19_statefulpartitionedcall_args_19%vgg19_statefulpartitionedcall_args_20%vgg19_statefulpartitionedcall_args_21%vgg19_statefulpartitionedcall_args_22%vgg19_statefulpartitionedcall_args_23%vgg19_statefulpartitionedcall_args_24%vgg19_statefulpartitionedcall_args_25%vgg19_statefulpartitionedcall_args_26%vgg19_statefulpartitionedcall_args_27%vgg19_statefulpartitionedcall_args_28%vgg19_statefulpartitionedcall_args_29%vgg19_statefulpartitionedcall_args_30%vgg19_statefulpartitionedcall_args_31%vgg19_statefulpartitionedcall_args_32*-
_gradient_op_typePartitionedCall-274298*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_274257*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*,
Tin%
#2!Ќ
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-274379*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_274373*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2«
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2*-
_gradient_op_typePartitionedCall-274425*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_274413v
	tags/CastCast dropout/PartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒИ
tags/StatefulPartitionedCallStatefulPartitionedCalltags/Cast:y:0#tags_statefulpartitionedcall_args_1#tags_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*'
_output_shapes
:€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-274448*I
fDRB
@__inference_tags_layer_call_and_return_conditional_losses_274442*
Tout
2u
cc3/CastCast dropout/PartitionedCall:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒГ
cc3/StatefulPartitionedCallStatefulPartitionedCallcc3/Cast:y:0"cc3_statefulpartitionedcall_args_1"cc3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-274477*H
fCRA
?__inference_cc3_layer_call_and_return_conditional_losses_274471*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*'
_output_shapes
:€€€€€€€€€*
Tin
2…
IdentityIdentity$cc3/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ћ

Identity_1Identity%tags/StatefulPartitionedCall:output:0^cc3/StatefulPartitionedCall^tags/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2<
tags/StatefulPartitionedCalltags/StatefulPartitionedCall2:
cc3/StatefulPartitionedCallcc3/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ 
 
D
(__inference_flatten_layer_call_fn_275812

inputs
identity•
PartitionedCallPartitionedCallinputs*4
config_proto$"

CPU

GPU2*0J 8RRЄ*)
_output_shapes
:€€€€€€€€€Аƒ*
Tin
2*-
_gradient_op_typePartitionedCall-274379*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_274373*
Tout
2b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:& "
 
_user_specified_nameinputs
оl
с
A__inference_vgg19_layer_call_and_return_conditional_losses_273758
input_1/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block3_conv4_statefulpartitionedcall_args_1/
+block3_conv4_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block4_conv4_statefulpartitionedcall_args_1/
+block4_conv4_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2/
+block5_conv4_statefulpartitionedcall_args_1/
+block5_conv4_statefulpartitionedcall_args_2
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCallђ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273233*Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*1
_output_shapes
:€€€€€€€€€аа@“
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*1
_output_shapes
:€€€€€€€€€аа@*
Tin
2*-
_gradient_op_typePartitionedCall-273258в
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273277*P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*/
_output_shapes
:€€€€€€€€€pp@»
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273300*Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€ppА—
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273325*Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€ppА*
Tin
2г
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273344*P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А»
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273367*Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361—
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273392*Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€88А*
Tin
2—
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*0
_output_shapes
:€€€€€€€€€88А*
Tin
2*-
_gradient_op_typePartitionedCall-273417*Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0+block3_conv4_statefulpartitionedcall_args_1+block3_conv4_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273442*Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄг
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273461*P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ»
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273484*Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2—
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273509*Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503—
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273534*Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А—
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0+block4_conv4_statefulpartitionedcall_args_1+block4_conv4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273559*Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€Аг
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273578*P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А»
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273601*Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595—
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273626*Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273651*Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645*
Tout
2—
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0+block5_conv4_statefulpartitionedcall_args_1+block5_conv4_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273676*Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670*
Tout
2г
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273695*P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2е
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  
Ґ
c
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338

inputs
identityЂ
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
л
ц	
&__inference_vgg19_layer_call_fn_275446

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identityИҐStatefulPartitionedCallс

StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_273877*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*,
Tin%
#2!*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273878Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:€€€€€€€€€А*
T0"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :  :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
ѓ
H
,__inference_block2_pool_layer_call_fn_273347

inputs
identity 
PartitionedCallPartitionedCallinputs*P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tin
2*-
_gradient_op_typePartitionedCall-273344Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
П
б
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0*
strides
*
paddingSAME°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
П
б
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
лl
р
A__inference_vgg19_layer_call_and_return_conditional_losses_273974

inputs/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block3_conv4_statefulpartitionedcall_args_1/
+block3_conv4_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block4_conv4_statefulpartitionedcall_args_1/
+block4_conv4_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2/
+block5_conv4_statefulpartitionedcall_args_1/
+block5_conv4_statefulpartitionedcall_args_2
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCallЂ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tin
2*1
_output_shapes
:€€€€€€€€€аа@*-
_gradient_op_typePartitionedCall-273233*Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ“
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*1
_output_shapes
:€€€€€€€€€аа@*-
_gradient_op_typePartitionedCall-273258*Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252*
Tout
2в
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:€€€€€€€€€pp@*-
_gradient_op_typePartitionedCall-273277*P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ»
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€ppА*
Tin
2*-
_gradient_op_typePartitionedCall-273300*Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294—
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273325*Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€ppАг
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273344*P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338*
Tout
2»
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А*-
_gradient_op_typePartitionedCall-273367—
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273392*Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А—
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273417*Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88А—
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0+block3_conv4_statefulpartitionedcall_args_1+block3_conv4_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273442*Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€88Аг
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273461*P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455*
Tout
2»
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273484—
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273509—
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273534*Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528*
Tout
2—
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0+block4_conv4_statefulpartitionedcall_args_1+block4_conv4_statefulpartitionedcall_args_2*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273559*Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄг
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273578*P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572»
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273601*Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595—
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273626*Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ—
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*0
_output_shapes
:€€€€€€€€€А*
Tin
2*-
_gradient_op_typePartitionedCall-273651*Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645—
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0+block5_conv4_statefulpartitionedcall_args_1+block5_conv4_statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273676*Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670*
Tout
2г
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-273695*P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*0
_output_shapes
:€€€€€€€€€Ае
IdentityIdentity$block5_pool/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*≤
_input_shapes†
Э:€€€€€€€€€аа::::::::::::::::::::::::::::::::2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall: : : : : : : : :  :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : 
ї
Ѓ
-__inference_block2_conv2_layer_call_fn_273330

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273325*Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Ух
ч
!__inference__wrapped_model_273213	
input;
7model_vgg19_block1_conv1_conv2d_readvariableop_resource<
8model_vgg19_block1_conv1_biasadd_readvariableop_resource;
7model_vgg19_block1_conv2_conv2d_readvariableop_resource<
8model_vgg19_block1_conv2_biasadd_readvariableop_resource;
7model_vgg19_block2_conv1_conv2d_readvariableop_resource<
8model_vgg19_block2_conv1_biasadd_readvariableop_resource;
7model_vgg19_block2_conv2_conv2d_readvariableop_resource<
8model_vgg19_block2_conv2_biasadd_readvariableop_resource;
7model_vgg19_block3_conv1_conv2d_readvariableop_resource<
8model_vgg19_block3_conv1_biasadd_readvariableop_resource;
7model_vgg19_block3_conv2_conv2d_readvariableop_resource<
8model_vgg19_block3_conv2_biasadd_readvariableop_resource;
7model_vgg19_block3_conv3_conv2d_readvariableop_resource<
8model_vgg19_block3_conv3_biasadd_readvariableop_resource;
7model_vgg19_block3_conv4_conv2d_readvariableop_resource<
8model_vgg19_block3_conv4_biasadd_readvariableop_resource;
7model_vgg19_block4_conv1_conv2d_readvariableop_resource<
8model_vgg19_block4_conv1_biasadd_readvariableop_resource;
7model_vgg19_block4_conv2_conv2d_readvariableop_resource<
8model_vgg19_block4_conv2_biasadd_readvariableop_resource;
7model_vgg19_block4_conv3_conv2d_readvariableop_resource<
8model_vgg19_block4_conv3_biasadd_readvariableop_resource;
7model_vgg19_block4_conv4_conv2d_readvariableop_resource<
8model_vgg19_block4_conv4_biasadd_readvariableop_resource;
7model_vgg19_block5_conv1_conv2d_readvariableop_resource<
8model_vgg19_block5_conv1_biasadd_readvariableop_resource;
7model_vgg19_block5_conv2_conv2d_readvariableop_resource<
8model_vgg19_block5_conv2_biasadd_readvariableop_resource;
7model_vgg19_block5_conv3_conv2d_readvariableop_resource<
8model_vgg19_block5_conv3_biasadd_readvariableop_resource;
7model_vgg19_block5_conv4_conv2d_readvariableop_resource<
8model_vgg19_block5_conv4_biasadd_readvariableop_resource-
)model_tags_matmul_readvariableop_resource.
*model_tags_biasadd_readvariableop_resource,
(model_cc3_matmul_readvariableop_resource-
)model_cc3_biasadd_readvariableop_resource
identity

identity_1ИҐ model/cc3/BiasAdd/ReadVariableOpҐmodel/cc3/MatMul/ReadVariableOpҐ!model/tags/BiasAdd/ReadVariableOpҐ model/tags/MatMul/ReadVariableOpҐ/model/vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ.model/vgg19/block1_conv1/Conv2D/ReadVariableOpҐ/model/vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ.model/vgg19/block1_conv2/Conv2D/ReadVariableOpҐ/model/vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ.model/vgg19/block2_conv1/Conv2D/ReadVariableOpҐ/model/vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ.model/vgg19/block2_conv2/Conv2D/ReadVariableOpҐ/model/vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ.model/vgg19/block3_conv1/Conv2D/ReadVariableOpҐ/model/vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ.model/vgg19/block3_conv2/Conv2D/ReadVariableOpҐ/model/vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ.model/vgg19/block3_conv3/Conv2D/ReadVariableOpҐ/model/vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ.model/vgg19/block3_conv4/Conv2D/ReadVariableOpҐ/model/vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ.model/vgg19/block4_conv1/Conv2D/ReadVariableOpҐ/model/vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ.model/vgg19/block4_conv2/Conv2D/ReadVariableOpҐ/model/vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ.model/vgg19/block4_conv3/Conv2D/ReadVariableOpҐ/model/vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ.model/vgg19/block4_conv4/Conv2D/ReadVariableOpҐ/model/vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ.model/vgg19/block5_conv1/Conv2D/ReadVariableOpҐ/model/vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ.model/vgg19/block5_conv2/Conv2D/ReadVariableOpҐ/model/vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ.model/vgg19/block5_conv3/Conv2D/ReadVariableOpҐ/model/vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ.model/vgg19/block5_conv4/Conv2D/ReadVariableOpw
model/vgg19/block1_conv1/CastCastinput*

SrcT0*

DstT0*1
_output_shapes
:€€€€€€€€€аа№
.model/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@и
model/vgg19/block1_conv1/Conv2DConv2D!model/vgg19/block1_conv1/Cast:y:06model/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
“
/model/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ 
 model/vgg19/block1_conv1/BiasAddBiasAdd(model/vgg19/block1_conv1/Conv2D:output:07model/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@М
model/vgg19/block1_conv1/ReluRelu)model/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@№
.model/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@т
model/vgg19/block1_conv2/Conv2DConv2D+model/vgg19/block1_conv1/Relu:activations:06model/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:€€€€€€€€€аа@*
T0*
strides
*
paddingSAME“
/model/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ 
 model/vgg19/block1_conv2/BiasAddBiasAdd(model/vgg19/block1_conv2/Conv2D:output:07model/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@М
model/vgg19/block1_conv2/ReluRelu)model/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@Ќ
model/vgg19/block1_pool/MaxPoolMaxPool+model/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
T0*
strides
*
ksize
*
paddingVALIDЁ
.model/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@Ао
model/vgg19/block2_conv1/Conv2DConv2D(model/vgg19/block1_pool/MaxPool:output:06model/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА”
/model/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block2_conv1/BiasAddBiasAdd(model/vgg19/block2_conv1/Conv2D:output:07model/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАЛ
model/vgg19/block2_conv1/ReluRelu)model/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАё
.model/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block2_conv2/Conv2DConv2D+model/vgg19/block2_conv1/Relu:activations:06model/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€ppА*
T0*
strides
”
/model/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block2_conv2/BiasAddBiasAdd(model/vgg19/block2_conv2/Conv2D:output:07model/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppАЛ
model/vgg19/block2_conv2/ReluRelu)model/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppАќ
model/vgg19/block2_pool/MaxPoolMaxPool+model/vgg19/block2_conv2/Relu:activations:0*
T0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€88Аё
.model/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААо
model/vgg19/block3_conv1/Conv2DConv2D(model/vgg19/block2_pool/MaxPool:output:06model/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0”
/model/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block3_conv1/BiasAddBiasAdd(model/vgg19/block3_conv1/Conv2D:output:07model/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88АЛ
model/vgg19/block3_conv1/ReluRelu)model/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аё
.model/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block3_conv2/Conv2DConv2D+model/vgg19/block3_conv1/Relu:activations:06model/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А*
T0*
strides
”
/model/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block3_conv2/BiasAddBiasAdd(model/vgg19/block3_conv2/Conv2D:output:07model/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88АЛ
model/vgg19/block3_conv2/ReluRelu)model/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аё
.model/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block3_conv3/Conv2DConv2D+model/vgg19/block3_conv2/Relu:activations:06model/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А”
/model/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block3_conv3/BiasAddBiasAdd(model/vgg19/block3_conv3/Conv2D:output:07model/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88АЛ
model/vgg19/block3_conv3/ReluRelu)model/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аё
.model/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block3_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block3_conv4/Conv2DConv2D+model/vgg19/block3_conv3/Relu:activations:06model/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€88А”
/model/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block3_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block3_conv4/BiasAddBiasAdd(model/vgg19/block3_conv4/Conv2D:output:07model/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88АЛ
model/vgg19/block3_conv4/ReluRelu)model/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88Аќ
model/vgg19/block3_pool/MaxPoolMaxPool+model/vgg19/block3_conv4/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
ё
.model/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААо
model/vgg19/block4_conv1/Conv2DConv2D(model/vgg19/block3_pool/MaxPool:output:06model/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А”
/model/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block4_conv1/BiasAddBiasAdd(model/vgg19/block4_conv1/Conv2D:output:07model/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block4_conv1/ReluRelu)model/vgg19/block4_conv1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А*
T0ё
.model/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block4_conv2/Conv2DConv2D+model/vgg19/block4_conv1/Relu:activations:06model/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAME”
/model/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block4_conv2/BiasAddBiasAdd(model/vgg19/block4_conv2/Conv2D:output:07model/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block4_conv2/ReluRelu)model/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аё
.model/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block4_conv3/Conv2DConv2D+model/vgg19/block4_conv2/Relu:activations:06model/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0”
/model/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block4_conv3/BiasAddBiasAdd(model/vgg19/block4_conv3/Conv2D:output:07model/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block4_conv3/ReluRelu)model/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аё
.model/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block4_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block4_conv4/Conv2DConv2D+model/vgg19/block4_conv3/Relu:activations:06model/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAME”
/model/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block4_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block4_conv4/BiasAddBiasAdd(model/vgg19/block4_conv4/Conv2D:output:07model/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block4_conv4/ReluRelu)model/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аќ
model/vgg19/block4_pool/MaxPoolMaxPool+model/vgg19/block4_conv4/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:€€€€€€€€€А*
T0ё
.model/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААо
model/vgg19/block5_conv1/Conv2DConv2D(model/vgg19/block4_pool/MaxPool:output:06model/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
”
/model/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block5_conv1/BiasAddBiasAdd(model/vgg19/block5_conv1/Conv2D:output:07model/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block5_conv1/ReluRelu)model/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аё
.model/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block5_conv2/Conv2DConv2D+model/vgg19/block5_conv1/Relu:activations:06model/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А”
/model/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block5_conv2/BiasAddBiasAdd(model/vgg19/block5_conv2/Conv2D:output:07model/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block5_conv2/ReluRelu)model/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аё
.model/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block5_conv3/Conv2DConv2D+model/vgg19/block5_conv2/Relu:activations:06model/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:€€€€€€€€€А”
/model/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block5_conv3/BiasAddBiasAdd(model/vgg19/block5_conv3/Conv2D:output:07model/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
model/vgg19/block5_conv3/ReluRelu)model/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аё
.model/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp7model_vgg19_block5_conv4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААс
model/vgg19/block5_conv4/Conv2DConv2D+model/vgg19/block5_conv3/Relu:activations:06model/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
paddingSAME”
/model/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp8model_vgg19_block5_conv4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:А…
 model/vgg19/block5_conv4/BiasAddBiasAdd(model/vgg19/block5_conv4/Conv2D:output:07model/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:€€€€€€€€€А*
T0Л
model/vgg19/block5_conv4/ReluRelu)model/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аќ
model/vgg19/block5_pool/MaxPoolMaxPool+model/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
T0*
strides
*
ksize
*
paddingVALIDl
model/flatten/Reshape/shapeConst*
valueB"€€€€ b  *
dtype0*
_output_shapes
:§
model/flatten/ReshapeReshape(model/vgg19/block5_pool/MaxPool:output:0$model/flatten/Reshape/shape:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒv
model/dropout/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*)
_output_shapes
:€€€€€€€€€Аƒ{
model/tags/CastCastmodel/dropout/Identity:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒЇ
 model/tags/MatMul/ReadVariableOpReadVariableOp)model_tags_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
АƒМ
model/tags/MatMulMatMulmodel/tags/Cast:y:0(model/tags/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
!model/tags/BiasAdd/ReadVariableOpReadVariableOp*model_tags_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Ч
model/tags/BiasAddBiasAddmodel/tags/MatMul:product:0)model/tags/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€l
model/tags/SigmoidSigmoidmodel/tags/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
model/cc3/CastCastmodel/dropout/Identity:output:0*

SrcT0*

DstT0*)
_output_shapes
:€€€€€€€€€АƒЄ
model/cc3/MatMul/ReadVariableOpReadVariableOp(model_cc3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
АƒЙ
model/cc3/MatMulMatMulmodel/cc3/Cast:y:0'model/cc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€і
 model/cc3/BiasAdd/ReadVariableOpReadVariableOp)model_cc3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Ф
model/cc3/BiasAddBiasAddmodel/cc3/MatMul:product:0(model/cc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€j
model/cc3/SoftmaxSoftmaxmodel/cc3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Я
IdentityIdentitymodel/cc3/Softmax:softmax:0!^model/cc3/BiasAdd/ReadVariableOp ^model/cc3/MatMul/ReadVariableOp"^model/tags/BiasAdd/ReadVariableOp!^model/tags/MatMul/ReadVariableOp0^model/vgg19/block1_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block1_conv1/Conv2D/ReadVariableOp0^model/vgg19/block1_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block1_conv2/Conv2D/ReadVariableOp0^model/vgg19/block2_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block2_conv1/Conv2D/ReadVariableOp0^model/vgg19/block2_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block2_conv2/Conv2D/ReadVariableOp0^model/vgg19/block3_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv1/Conv2D/ReadVariableOp0^model/vgg19/block3_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv2/Conv2D/ReadVariableOp0^model/vgg19/block3_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv3/Conv2D/ReadVariableOp0^model/vgg19/block3_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv4/Conv2D/ReadVariableOp0^model/vgg19/block4_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv1/Conv2D/ReadVariableOp0^model/vgg19/block4_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv2/Conv2D/ReadVariableOp0^model/vgg19/block4_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv3/Conv2D/ReadVariableOp0^model/vgg19/block4_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv4/Conv2D/ReadVariableOp0^model/vgg19/block5_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv1/Conv2D/ReadVariableOp0^model/vgg19/block5_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv2/Conv2D/ReadVariableOp0^model/vgg19/block5_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv3/Conv2D/ReadVariableOp0^model/vgg19/block5_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv4/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€Ь

Identity_1Identitymodel/tags/Sigmoid:y:0!^model/cc3/BiasAdd/ReadVariableOp ^model/cc3/MatMul/ReadVariableOp"^model/tags/BiasAdd/ReadVariableOp!^model/tags/MatMul/ReadVariableOp0^model/vgg19/block1_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block1_conv1/Conv2D/ReadVariableOp0^model/vgg19/block1_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block1_conv2/Conv2D/ReadVariableOp0^model/vgg19/block2_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block2_conv1/Conv2D/ReadVariableOp0^model/vgg19/block2_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block2_conv2/Conv2D/ReadVariableOp0^model/vgg19/block3_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv1/Conv2D/ReadVariableOp0^model/vgg19/block3_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv2/Conv2D/ReadVariableOp0^model/vgg19/block3_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv3/Conv2D/ReadVariableOp0^model/vgg19/block3_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block3_conv4/Conv2D/ReadVariableOp0^model/vgg19/block4_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv1/Conv2D/ReadVariableOp0^model/vgg19/block4_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv2/Conv2D/ReadVariableOp0^model/vgg19/block4_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv3/Conv2D/ReadVariableOp0^model/vgg19/block4_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block4_conv4/Conv2D/ReadVariableOp0^model/vgg19/block5_conv1/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv1/Conv2D/ReadVariableOp0^model/vgg19/block5_conv2/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv2/Conv2D/ReadVariableOp0^model/vgg19/block5_conv3/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv3/Conv2D/ReadVariableOp0^model/vgg19/block5_conv4/BiasAdd/ReadVariableOp/^model/vgg19/block5_conv4/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::2b
/model/vgg19/block5_conv4/BiasAdd/ReadVariableOp/model/vgg19/block5_conv4/BiasAdd/ReadVariableOp2`
.model/vgg19/block4_conv1/Conv2D/ReadVariableOp.model/vgg19/block4_conv1/Conv2D/ReadVariableOp2`
.model/vgg19/block3_conv3/Conv2D/ReadVariableOp.model/vgg19/block3_conv3/Conv2D/ReadVariableOp2B
model/cc3/MatMul/ReadVariableOpmodel/cc3/MatMul/ReadVariableOp2`
.model/vgg19/block4_conv2/Conv2D/ReadVariableOp.model/vgg19/block4_conv2/Conv2D/ReadVariableOp2`
.model/vgg19/block3_conv4/Conv2D/ReadVariableOp.model/vgg19/block3_conv4/Conv2D/ReadVariableOp2`
.model/vgg19/block1_conv1/Conv2D/ReadVariableOp.model/vgg19/block1_conv1/Conv2D/ReadVariableOp2b
/model/vgg19/block3_conv1/BiasAdd/ReadVariableOp/model/vgg19/block3_conv1/BiasAdd/ReadVariableOp2`
.model/vgg19/block5_conv1/Conv2D/ReadVariableOp.model/vgg19/block5_conv1/Conv2D/ReadVariableOp2b
/model/vgg19/block4_conv2/BiasAdd/ReadVariableOp/model/vgg19/block4_conv2/BiasAdd/ReadVariableOp2`
.model/vgg19/block4_conv3/Conv2D/ReadVariableOp.model/vgg19/block4_conv3/Conv2D/ReadVariableOp2b
/model/vgg19/block1_conv2/BiasAdd/ReadVariableOp/model/vgg19/block1_conv2/BiasAdd/ReadVariableOp2b
/model/vgg19/block5_conv3/BiasAdd/ReadVariableOp/model/vgg19/block5_conv3/BiasAdd/ReadVariableOp2`
.model/vgg19/block1_conv2/Conv2D/ReadVariableOp.model/vgg19/block1_conv2/Conv2D/ReadVariableOp2b
/model/vgg19/block3_conv4/BiasAdd/ReadVariableOp/model/vgg19/block3_conv4/BiasAdd/ReadVariableOp2`
.model/vgg19/block5_conv2/Conv2D/ReadVariableOp.model/vgg19/block5_conv2/Conv2D/ReadVariableOp2`
.model/vgg19/block4_conv4/Conv2D/ReadVariableOp.model/vgg19/block4_conv4/Conv2D/ReadVariableOp2`
.model/vgg19/block2_conv1/Conv2D/ReadVariableOp.model/vgg19/block2_conv1/Conv2D/ReadVariableOp2`
.model/vgg19/block5_conv3/Conv2D/ReadVariableOp.model/vgg19/block5_conv3/Conv2D/ReadVariableOp2b
/model/vgg19/block4_conv1/BiasAdd/ReadVariableOp/model/vgg19/block4_conv1/BiasAdd/ReadVariableOp2b
/model/vgg19/block1_conv1/BiasAdd/ReadVariableOp/model/vgg19/block1_conv1/BiasAdd/ReadVariableOp2b
/model/vgg19/block5_conv2/BiasAdd/ReadVariableOp/model/vgg19/block5_conv2/BiasAdd/ReadVariableOp2`
.model/vgg19/block2_conv2/Conv2D/ReadVariableOp.model/vgg19/block2_conv2/Conv2D/ReadVariableOp2b
/model/vgg19/block2_conv2/BiasAdd/ReadVariableOp/model/vgg19/block2_conv2/BiasAdd/ReadVariableOp2F
!model/tags/BiasAdd/ReadVariableOp!model/tags/BiasAdd/ReadVariableOp2D
 model/tags/MatMul/ReadVariableOp model/tags/MatMul/ReadVariableOp2b
/model/vgg19/block3_conv3/BiasAdd/ReadVariableOp/model/vgg19/block3_conv3/BiasAdd/ReadVariableOp2D
 model/cc3/BiasAdd/ReadVariableOp model/cc3/BiasAdd/ReadVariableOp2b
/model/vgg19/block4_conv4/BiasAdd/ReadVariableOp/model/vgg19/block4_conv4/BiasAdd/ReadVariableOp2`
.model/vgg19/block5_conv4/Conv2D/ReadVariableOp.model/vgg19/block5_conv4/Conv2D/ReadVariableOp2`
.model/vgg19/block3_conv1/Conv2D/ReadVariableOp.model/vgg19/block3_conv1/Conv2D/ReadVariableOp2`
.model/vgg19/block3_conv2/Conv2D/ReadVariableOp.model/vgg19/block3_conv2/Conv2D/ReadVariableOp2b
/model/vgg19/block5_conv1/BiasAdd/ReadVariableOp/model/vgg19/block5_conv1/BiasAdd/ReadVariableOp2b
/model/vgg19/block2_conv1/BiasAdd/ReadVariableOp/model/vgg19/block2_conv1/BiasAdd/ReadVariableOp2b
/model/vgg19/block3_conv2/BiasAdd/ReadVariableOp/model/vgg19/block3_conv2/BiasAdd/ReadVariableOp2b
/model/vgg19/block4_conv3/BiasAdd/ReadVariableOp/model/vgg19/block4_conv3/BiasAdd/ReadVariableOp:$ :% !

_user_specified_nameinput: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# 
П
б
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ї
Ѓ
-__inference_block4_conv1_layer_call_fn_273489

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273484*Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
Tin
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
ї
Ѓ
-__inference_block3_conv1_layer_call_fn_273372

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273367*Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361*
Tout
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
л
Ъ
&__inference_model_layer_call_fn_275167

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36
identity

identity_1ИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36*4
config_proto$"

CPU

GPU2*0J 8RRЄ*:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*0
Tin)
'2%*-
_gradient_op_typePartitionedCall-274686*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_274685*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€Д

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*¬
_input_shapes∞
≠:€€€€€€€€€аа::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :& "
 
_user_specified_nameinputs: : : : : 
ї
Ѓ
-__inference_block3_conv3_layer_call_fn_273422

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-273417*Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
П
б
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
З
б
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp™
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
T0*
strides
†
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@•
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ѓ
H
,__inference_block3_pool_layer_call_fn_273464

inputs
identity 
PartitionedCallPartitionedCallinputs*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*-
_gradient_op_typePartitionedCall-273461*P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455*
Tout
2Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
ї
Ѓ
-__inference_block4_conv4_layer_call_fn_273564

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*4
config_proto$"

CPU

GPU2*0J 8RRЄ*
Tin
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*-
_gradient_op_typePartitionedCall-273559*Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Ґ
c
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
П
б
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpђ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ААђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А°
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*ж
serving_default“
A
input8
serving_default_input:0€€€€€€€€€аа8
tags0
StatefulPartitionedCall:1€€€€€€€€€7
cc30
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:цц
…ў
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
+–&call_and_return_all_conditional_losses
—_default_save_signature
“__call__"ё÷
_tf_keras_model√÷{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "vgg19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float16", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg19", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float16", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["vgg19", 1, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float16", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cc3", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cc3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "tags", "trainable": true, "dtype": "float32", "units": 3, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "tags", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["cc3", 0, 0], ["tags", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "vgg19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float16", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "name": "vgg19", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float16", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["vgg19", 1, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float16", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "cc3", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "cc3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "tags", "trainable": true, "dtype": "float32", "units": 3, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "tags", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["cc3", 0, 0], ["tags", 0, 0]]}}, "training_config": {"loss": {"cc3": "categorical_crossentropy", "tags": "binary_crossentropy"}, "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": {"cc3": 1.0, "tags": 1.0}, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.9999999494757503e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
≥
	variables
regularization_losses
trainable_variables
	keras_api
+”&call_and_return_all_conditional_losses
‘__call__"Ґ
_tf_keras_layerИ{"class_name": "InputLayer", "name": "input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 224, 224, 3], "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input"}}
№Њ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
 layer_with_weights-11
 layer-15
!layer-16
"layer_with_weights-12
"layer-17
#layer_with_weights-13
#layer-18
$layer_with_weights-14
$layer-19
%layer_with_weights-15
%layer-20
&layer-21
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+’&call_and_return_all_conditional_losses
÷__call__"ъЈ
_tf_keras_modelяЈ{"class_name": "Model", "name": "vgg19", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": null, "config": {"name": "vgg19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float16", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "vgg19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float16", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv4", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv4", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv4", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv4", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv4", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["block5_pool", 0, 0]]}}}
Ѓ
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+„&call_and_return_all_conditional_losses
Ў__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float16", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ѓ
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+ў&call_and_return_all_conditional_losses
Џ__call__"Э
_tf_keras_layerГ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float16", "rate": 0.25, "noise_shape": null, "seed": null}}
р

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
+џ&call_and_return_all_conditional_losses
№__call__"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "cc3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cc3", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25088}}}}
т

9kernel
:bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "tags", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tags", "trainable": true, "dtype": "float32", "units": 3, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25088}}}}
£
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate3m»4m…9m :mЋ3vћ4vЌ9vќ:vѕ"
	optimizer
ґ
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31
332
433
934
:35"
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
92
:3"
trackable_list_wrapper
ї
dmetrics

elayers
	variables
	regularization_losses
fnon_trainable_variables

trainable_variables
glayer_regularization_losses
“__call__
—_default_save_signature
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
-
яserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
hmetrics

ilayers
	variables
regularization_losses
jnon_trainable_variables
trainable_variables
klayer_regularization_losses
‘__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
Ј
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+а&call_and_return_all_conditional_losses
б__call__"¶
_tf_keras_layerМ{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": [null, 224, 224, 3], "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float16", "sparse": false, "name": "input_1"}}
ч

Dkernel
Ebias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+в&call_and_return_all_conditional_losses
г__call__"–
_tf_keras_layerґ{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
ш

Fkernel
Gbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+д&call_and_return_all_conditional_losses
е__call__"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float16", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
щ
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"и
_tf_keras_layerќ{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block1_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ

Hkernel
Ibias
|	variables
}regularization_losses
~trainable_variables
	keras_api
+и&call_and_return_all_conditional_losses
й__call__"“
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ю

Jkernel
Kbias
А	variables
Бregularization_losses
Вtrainable_variables
Г	keras_api
+к&call_and_return_all_conditional_losses
л__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float16", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
э
Д	variables
Еregularization_losses
Жtrainable_variables
З	keras_api
+м&call_and_return_all_conditional_losses
н__call__"и
_tf_keras_layerќ{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block2_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю

Lkernel
Mbias
И	variables
Йregularization_losses
Кtrainable_variables
Л	keras_api
+о&call_and_return_all_conditional_losses
п__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block3_conv1", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ю

Nkernel
Obias
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
+р&call_and_return_all_conditional_losses
с__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block3_conv2", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
ю

Pkernel
Qbias
Р	variables
Сregularization_losses
Тtrainable_variables
У	keras_api
+т&call_and_return_all_conditional_losses
у__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block3_conv3", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
ю

Rkernel
Sbias
Ф	variables
Хregularization_losses
Цtrainable_variables
Ч	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block3_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block3_conv4", "trainable": false, "dtype": "float16", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
э
Ш	variables
Щregularization_losses
Ъtrainable_variables
Ы	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"и
_tf_keras_layerќ{"class_name": "MaxPooling2D", "name": "block3_pool", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block3_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю

Tkernel
Ubias
Ь	variables
Эregularization_losses
Юtrainable_variables
Я	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block4_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
ю

Vkernel
Wbias
†	variables
°regularization_losses
Ґtrainable_variables
£	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block4_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
ю

Xkernel
Ybias
§	variables
•regularization_losses
¶trainable_variables
І	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block4_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
ю

Zkernel
[bias
®	variables
©regularization_losses
™trainable_variables
Ђ	keras_api
+ю&call_and_return_all_conditional_losses
€__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block4_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block4_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
э
ђ	variables
≠regularization_losses
Ѓtrainable_variables
ѓ	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"и
_tf_keras_layerќ{"class_name": "MaxPooling2D", "name": "block4_pool", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block4_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю

\kernel
]bias
∞	variables
±regularization_losses
≤trainable_variables
≥	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block5_conv1", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
ю

^kernel
_bias
і	variables
µregularization_losses
ґtrainable_variables
Ј	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block5_conv2", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
ю

`kernel
abias
Є	variables
єregularization_losses
Їtrainable_variables
ї	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block5_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block5_conv3", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
ю

bkernel
cbias
Љ	variables
љregularization_losses
Њtrainable_variables
њ	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"”
_tf_keras_layerє{"class_name": "Conv2D", "name": "block5_conv4", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block5_conv4", "trainable": false, "dtype": "float16", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}}
э
ј	variables
Ѕregularization_losses
¬trainable_variables
√	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"и
_tf_keras_layerќ{"class_name": "MaxPooling2D", "name": "block5_pool", "trainable": false, "expects_training_arg": false, "dtype": "float16", "batch_input_shape": null, "config": {"name": "block5_pool", "trainable": false, "dtype": "float16", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ƒmetrics
≈layers
'	variables
(regularization_losses
∆non_trainable_variables
)trainable_variables
 «layer_regularization_losses
÷__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
»metrics
…layers
+	variables
,regularization_losses
 non_trainable_variables
-trainable_variables
 Ћlayer_regularization_losses
Ў__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ћmetrics
Ќlayers
/	variables
0regularization_losses
ќnon_trainable_variables
1trainable_variables
 ѕlayer_regularization_losses
Џ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 :
Аƒ2cc3_2/kernel
:2
cc3_2/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
–metrics
—layers
5	variables
6regularization_losses
“non_trainable_variables
7trainable_variables
 ”layer_regularization_losses
№__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
!:
Аƒ2tags_2/kernel
:2tags_2/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
°
‘metrics
’layers
;	variables
<regularization_losses
÷non_trainable_variables
=trainable_variables
 „layer_regularization_losses
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_1/Adam/iter
 : (2training_1/Adam/beta_1
 : (2training_1/Adam/beta_2
: (2training_1/Adam/decay
':% (2training_1/Adam/learning_rate
/:-@2block1_conv1_2/kernel
!:@2block1_conv1_2/bias
/:-@@2block1_conv2_2/kernel
!:@2block1_conv2_2/bias
0:.@А2block2_conv1_2/kernel
": А2block2_conv1_2/bias
1:/АА2block2_conv2_2/kernel
": А2block2_conv2_2/bias
1:/АА2block3_conv1_2/kernel
": А2block3_conv1_2/bias
1:/АА2block3_conv2_2/kernel
": А2block3_conv2_2/bias
1:/АА2block3_conv3_2/kernel
": А2block3_conv3_2/bias
1:/АА2block3_conv4_2/kernel
": А2block3_conv4_2/bias
1:/АА2block4_conv1_2/kernel
": А2block4_conv1_2/bias
1:/АА2block4_conv2_2/kernel
": А2block4_conv2_2/bias
1:/АА2block4_conv3_2/kernel
": А2block4_conv3_2/bias
1:/АА2block4_conv4_2/kernel
": А2block4_conv4_2/bias
1:/АА2block5_conv1_2/kernel
": А2block5_conv1_2/bias
1:/АА2block5_conv2_2/kernel
": А2block5_conv2_2/bias
1:/АА2block5_conv3_2/kernel
": А2block5_conv3_2/bias
1:/АА2block5_conv4_2/kernel
": А2block5_conv4_2/bias
0
Ў0
ў1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
Ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Џmetrics
џlayers
l	variables
mregularization_losses
№non_trainable_variables
ntrainable_variables
 Ёlayer_regularization_losses
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ёmetrics
яlayers
p	variables
qregularization_losses
аnon_trainable_variables
rtrainable_variables
 бlayer_regularization_losses
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
вmetrics
гlayers
t	variables
uregularization_losses
дnon_trainable_variables
vtrainable_variables
 еlayer_regularization_losses
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
жmetrics
зlayers
x	variables
yregularization_losses
иnon_trainable_variables
ztrainable_variables
 йlayer_regularization_losses
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
кmetrics
лlayers
|	variables
}regularization_losses
мnon_trainable_variables
~trainable_variables
 нlayer_regularization_losses
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
оmetrics
пlayers
А	variables
Бregularization_losses
рnon_trainable_variables
Вtrainable_variables
 сlayer_regularization_losses
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
тmetrics
уlayers
Д	variables
Еregularization_losses
фnon_trainable_variables
Жtrainable_variables
 хlayer_regularization_losses
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
цmetrics
чlayers
И	variables
Йregularization_losses
шnon_trainable_variables
Кtrainable_variables
 щlayer_regularization_losses
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ъmetrics
ыlayers
М	variables
Нregularization_losses
ьnon_trainable_variables
Оtrainable_variables
 эlayer_regularization_losses
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
юmetrics
€layers
Р	variables
Сregularization_losses
Аnon_trainable_variables
Тtrainable_variables
 Бlayer_regularization_losses
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Вmetrics
Гlayers
Ф	variables
Хregularization_losses
Дnon_trainable_variables
Цtrainable_variables
 Еlayer_regularization_losses
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Жmetrics
Зlayers
Ш	variables
Щregularization_losses
Иnon_trainable_variables
Ъtrainable_variables
 Йlayer_regularization_losses
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Кmetrics
Лlayers
Ь	variables
Эregularization_losses
Мnon_trainable_variables
Юtrainable_variables
 Нlayer_regularization_losses
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Оmetrics
Пlayers
†	variables
°regularization_losses
Рnon_trainable_variables
Ґtrainable_variables
 Сlayer_regularization_losses
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Тmetrics
Уlayers
§	variables
•regularization_losses
Фnon_trainable_variables
¶trainable_variables
 Хlayer_regularization_losses
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Цmetrics
Чlayers
®	variables
©regularization_losses
Шnon_trainable_variables
™trainable_variables
 Щlayer_regularization_losses
€__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ъmetrics
Ыlayers
ђ	variables
≠regularization_losses
Ьnon_trainable_variables
Ѓtrainable_variables
 Эlayer_regularization_losses
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Юmetrics
Яlayers
∞	variables
±regularization_losses
†non_trainable_variables
≤trainable_variables
 °layer_regularization_losses
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ґmetrics
£layers
і	variables
µregularization_losses
§non_trainable_variables
ґtrainable_variables
 •layer_regularization_losses
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
¶metrics
Іlayers
Є	variables
єregularization_losses
®non_trainable_variables
Їtrainable_variables
 ©layer_regularization_losses
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
™metrics
Ђlayers
Љ	variables
љregularization_losses
ђnon_trainable_variables
Њtrainable_variables
 ≠layer_regularization_losses
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ѓmetrics
ѓlayers
ј	variables
Ѕregularization_losses
∞non_trainable_variables
¬trainable_variables
 ±layer_regularization_losses
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
∆
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21"
trackable_list_wrapper
Ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
b30
c31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ђ

≤total

≥count
і
_fn_kwargs
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"н
_tf_keras_layer”{"class_name": "MeanMetricWrapper", "name": "cc3_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": null, "config": {"name": "cc3_accuracy", "dtype": "float16"}}
≠

єtotal

Їcount
ї
_fn_kwargs
Љ	variables
љregularization_losses
Њtrainable_variables
њ	keras_api
+О&call_and_return_all_conditional_losses
П__call__"п
_tf_keras_layer’{"class_name": "MeanMetricWrapper", "name": "tags_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float16", "batch_input_shape": null, "config": {"name": "tags_accuracy", "dtype": "float16"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
≤0
≥1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
јmetrics
Ѕlayers
µ	variables
ґregularization_losses
¬non_trainable_variables
Јtrainable_variables
 √layer_regularization_losses
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
є0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ƒmetrics
≈layers
Љ	variables
љregularization_losses
∆non_trainable_variables
Њtrainable_variables
 «layer_regularization_losses
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
≤0
≥1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
є0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
0:.
Аƒ2training_1/Adam/cc3_2/kernel/m
(:&2training_1/Adam/cc3_2/bias/m
1:/
Аƒ2training_1/Adam/tags_2/kernel/m
):'2training_1/Adam/tags_2/bias/m
0:.
Аƒ2training_1/Adam/cc3_2/kernel/v
(:&2training_1/Adam/cc3_2/bias/v
1:/
Аƒ2training_1/Adam/tags_2/kernel/v
):'2training_1/Adam/tags_2/bias/v
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_274540
A__inference_model_layer_call_and_return_conditional_losses_275081
A__inference_model_layer_call_and_return_conditional_losses_274490
A__inference_model_layer_call_and_return_conditional_losses_274939ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
!__inference__wrapped_model_273213Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input€€€€€€€€€аа
ж2г
&__inference_model_layer_call_fn_275167
&__inference_model_layer_call_fn_274727
&__inference_model_layer_call_fn_275124
&__inference_model_layer_call_fn_274633ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
Ў2’
A__inference_vgg19_layer_call_and_return_conditional_losses_275727
A__inference_vgg19_layer_call_and_return_conditional_losses_275288
A__inference_vgg19_layer_call_and_return_conditional_losses_275409
A__inference_vgg19_layer_call_and_return_conditional_losses_275605
A__inference_vgg19_layer_call_and_return_conditional_losses_273817
A__inference_vgg19_layer_call_and_return_conditional_losses_273758ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ґ2≥
&__inference_vgg19_layer_call_fn_275801
&__inference_vgg19_layer_call_fn_275764
&__inference_vgg19_layer_call_fn_275483
&__inference_vgg19_layer_call_fn_273913
&__inference_vgg19_layer_call_fn_274010
&__inference_vgg19_layer_call_fn_275446ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_275807Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_flatten_layer_call_fn_275812Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ƒ2Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_275832
C__inference_dropout_layer_call_and_return_conditional_losses_275837і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
(__inference_dropout_layer_call_fn_275847
(__inference_dropout_layer_call_fn_275842і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
й2ж
?__inference_cc3_layer_call_and_return_conditional_losses_275858Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
$__inference_cc3_layer_call_fn_275865Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_tags_layer_call_and_return_conditional_losses_275876Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_tags_layer_call_fn_275883Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
1B/
$__inference_signature_wrapper_274780input
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
І2§
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
М2Й
-__inference_block1_conv1_layer_call_fn_273238„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
І2§
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
М2Й
-__inference_block1_conv2_layer_call_fn_273263„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ѓ2ђ
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_block1_pool_layer_call_fn_273280а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
І2§
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
М2Й
-__inference_block2_conv1_layer_call_fn_273305„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
®2•
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block2_conv2_layer_call_fn_273330Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_block2_pool_layer_call_fn_273347а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block3_conv1_layer_call_fn_273372Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block3_conv2_layer_call_fn_273397Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block3_conv3_layer_call_fn_273422Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block3_conv4_layer_call_fn_273447Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_block3_pool_layer_call_fn_273464а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block4_conv1_layer_call_fn_273489Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block4_conv2_layer_call_fn_273514Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block4_conv3_layer_call_fn_273539Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block4_conv4_layer_call_fn_273564Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_block4_pool_layer_call_fn_273581а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block5_conv1_layer_call_fn_273606Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block5_conv2_layer_call_fn_273631Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block5_conv3_layer_call_fn_273656Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Н2К
-__inference_block5_conv4_layer_call_fn_273681Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_block5_pool_layer_call_fn_273698а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
(__inference_dropout_layer_call_fn_275847S5Ґ2
+Ґ(
"К
inputs€€€€€€€€€Аƒ
p 
™ "К€€€€€€€€€Аƒ“
&__inference_model_layer_call_fn_274633І$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34@Ґ=
6Ґ3
)К&
input€€€€€€€€€аа
p

 
™ "=Ъ:
К
0€€€€€€€€€
К
1€€€€€€€€€я
H__inference_block5_conv1_layer_call_and_return_conditional_losses_273595Т\]JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ я
H__inference_block5_conv3_layer_call_and_return_conditional_losses_273645Т`aJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
&__inference_vgg19_layer_call_fn_275801И DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "!К€€€€€€€€€Аџ
A__inference_vgg19_layer_call_and_return_conditional_losses_275409Х DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ µ
-__inference_block1_conv2_layer_call_fn_273263ГFGIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@µ
-__inference_block1_conv1_layer_call_fn_273238ГDEIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@z
%__inference_tags_layer_call_fn_275883Q9:1Ґ.
'Ґ$
"К
inputs€€€€€€€€€Аƒ
™ "К€€€€€€€€€≥
&__inference_vgg19_layer_call_fn_275764И DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "!К€€€€€€€€€Ая
H__inference_block2_conv2_layer_call_and_return_conditional_losses_273319ТJKJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ј
-__inference_block4_conv2_layer_call_fn_273514ЕVWJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А”
&__inference_model_layer_call_fn_275167®$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "=Ъ:
К
0€€€€€€€€€
К
1€€€€€€€€€Ј
-__inference_block4_conv4_layer_call_fn_273564ЕZ[JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block4_conv3_layer_call_fn_273539ЕXYJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ая
H__inference_block4_conv1_layer_call_and_return_conditional_losses_273478ТTUJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ “
&__inference_model_layer_call_fn_274727І$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34@Ґ=
6Ґ3
)К&
input€€€€€€€€€аа
p 

 
™ "=Ъ:
К
0€€€€€€€€€
К
1€€€€€€€€€я
H__inference_block4_conv3_layer_call_and_return_conditional_losses_273528ТXYJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ¬
,__inference_block2_pool_layer_call_fn_273347СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ґ
@__inference_tags_layer_call_and_return_conditional_losses_275876^9:1Ґ.
'Ґ$
"К
inputs€€€€€€€€€Аƒ
™ "%Ґ"
К
0€€€€€€€€€
Ъ В
(__inference_flatten_layer_call_fn_275812V8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€Аƒя
H__inference_block3_conv1_layer_call_and_return_conditional_losses_273361ТLMJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ¬
,__inference_block5_pool_layer_call_fn_273698СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€я
H__inference_block3_conv3_layer_call_and_return_conditional_losses_273411ТPQJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ј
-__inference_block4_conv1_layer_call_fn_273489ЕTUJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block2_conv2_layer_call_fn_273330ЕJKJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аі
&__inference_vgg19_layer_call_fn_273913Й DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcBҐ?
8Ґ5
+К(
input_1€€€€€€€€€аа
p

 
™ "!К€€€€€€€€€Аґ
-__inference_block2_conv1_layer_call_fn_273305ДHIIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А™
C__inference_flatten_layer_call_and_return_conditional_losses_275807c8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "'Ґ$
К
0€€€€€€€€€Аƒ
Ъ я
H__inference_block5_conv2_layer_call_and_return_conditional_losses_273620Т^_JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ь
A__inference_model_layer_call_and_return_conditional_losses_274939ґ$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "KҐH
AЪ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ к
G__inference_block1_pool_layer_call_and_return_conditional_losses_273271ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ °
?__inference_cc3_layer_call_and_return_conditional_losses_275858^341Ґ.
'Ґ$
"К
inputs€€€€€€€€€Аƒ
™ "%Ґ"
К
0€€€€€€€€€
Ъ к
G__inference_block2_pool_layer_call_and_return_conditional_losses_273338ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ џ
A__inference_vgg19_layer_call_and_return_conditional_losses_275605Х DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ І
C__inference_dropout_layer_call_and_return_conditional_losses_275832`5Ґ2
+Ґ(
"К
inputs€€€€€€€€€Аƒ
p
™ "'Ґ$
К
0€€€€€€€€€Аƒ
Ъ Ё
H__inference_block1_conv2_layer_call_and_return_conditional_losses_273252РFGIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≥
&__inference_vgg19_layer_call_fn_275446И DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "!К€€€€€€€€€АІ
C__inference_dropout_layer_call_and_return_conditional_losses_275837`5Ґ2
+Ґ(
"К
inputs€€€€€€€€€Аƒ
p 
™ "'Ґ$
К
0€€€€€€€€€Аƒ
Ъ і
&__inference_vgg19_layer_call_fn_274010Й DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcBҐ?
8Ґ5
+К(
input_1€€€€€€€€€аа
p 

 
™ "!К€€€€€€€€€Ак
G__inference_block3_pool_layer_call_and_return_conditional_losses_273455ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ я
H__inference_block4_conv2_layer_call_and_return_conditional_losses_273503ТVWJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ы
A__inference_model_layer_call_and_return_conditional_losses_274490µ$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34@Ґ=
6Ґ3
)К&
input€€€€€€€€€аа
p

 
™ "KҐH
AЪ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ ы
A__inference_model_layer_call_and_return_conditional_losses_274540µ$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34@Ґ=
6Ґ3
)К&
input€€€€€€€€€аа
p 

 
™ "KҐH
AЪ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ ¬
,__inference_block4_pool_layer_call_fn_273581СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
&__inference_vgg19_layer_call_fn_275483И DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "!К€€€€€€€€€Ая
H__inference_block5_conv4_layer_call_and_return_conditional_losses_273670ТbcJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ј
-__inference_block5_conv2_layer_call_fn_273631Е^_JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block5_conv1_layer_call_fn_273606Е\]JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block5_conv4_layer_call_fn_273681ЕbcJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block5_conv3_layer_call_fn_273656Е`aJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block3_conv3_layer_call_fn_273422ЕPQJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¬
,__inference_block1_pool_layer_call_fn_273280СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
!__inference__wrapped_model_273213≥$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:348Ґ5
.Ґ+
)К&
input€€€€€€€€€аа
™ "Q™N
&
tagsК
tags€€€€€€€€€
$
cc3К
cc3€€€€€€€€€Ј
-__inference_block3_conv4_layer_call_fn_273447ЕRSJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аё
H__inference_block2_conv1_layer_call_and_return_conditional_losses_273294СHIIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ џ
A__inference_vgg19_layer_call_and_return_conditional_losses_275727Х DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ к
G__inference_block4_pool_layer_call_and_return_conditional_losses_273572ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ь
A__inference_model_layer_call_and_return_conditional_losses_275081ґ$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "KҐH
AЪ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ y
$__inference_cc3_layer_call_fn_275865Q341Ґ.
'Ґ$
"К
inputs€€€€€€€€€Аƒ
™ "К€€€€€€€€€я
H__inference_block4_conv4_layer_call_and_return_conditional_losses_273553ТZ[JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ј
-__inference_block3_conv1_layer_call_fn_273372ЕLMJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЈ
-__inference_block3_conv2_layer_call_fn_273397ЕNOJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЁ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_273227РDEIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ №
A__inference_vgg19_layer_call_and_return_conditional_losses_273758Ц DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcBҐ?
8Ґ5
+К(
input_1€€€€€€€€€аа
p

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ¬
,__inference_block3_pool_layer_call_fn_273464СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€№
A__inference_vgg19_layer_call_and_return_conditional_losses_273817Ц DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcBҐ?
8Ґ5
+К(
input_1€€€€€€€€€аа
p 

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ к
G__inference_block5_pool_layer_call_and_return_conditional_losses_273689ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ я
H__inference_block3_conv2_layer_call_and_return_conditional_losses_273386ТNOJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
$__inference_signature_wrapper_274780Љ$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34AҐ>
Ґ 
7™4
2
input)К&
input€€€€€€€€€аа"Q™N
$
cc3К
cc3€€€€€€€€€
&
tagsК
tags€€€€€€€€€
(__inference_dropout_layer_call_fn_275842S5Ґ2
+Ґ(
"К
inputs€€€€€€€€€Аƒ
p
™ "К€€€€€€€€€Аƒя
H__inference_block3_conv4_layer_call_and_return_conditional_losses_273436ТRSJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ”
&__inference_model_layer_call_fn_275124®$DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abc9:34AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "=Ъ:
К
0€€€€€€€€€
К
1€€€€€€€€€џ
A__inference_vgg19_layer_call_and_return_conditional_losses_275288Х DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ 