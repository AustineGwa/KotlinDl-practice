
Z
default_data_placeholderPlaceholder*
dtype0*$
shape:?????????
:
ConstConst*
valueB"????  *
dtype0
j
dense_2_dense_kernel
VariableV2*
dtype0*
	container *
shape:
??*
shared_name 
c
dense_2_dense_bias
VariableV2*
shared_name *
dtype0*
	container *
shape:?
<
Const_1Const*
dtype0*
valueB"  ,  
D
Const_2Const*%
valueB	"               *
dtype0	
i
StatelessTruncatedNormalStatelessTruncatedNormalConst_1Const_2*
T0*
Tseed0	*
dtype0
8
Const_3Const*
valueB 2x?'>f??*
dtype0
=
CastCastConst_3*
Truncate( *

DstT0*

SrcT0
I
Init_dense_2_dense_kernelMulStatelessTruncatedNormalCast*
T0
?
Assign_dense_2_dense_kernelAssigndense_2_dense_kernelInit_dense_2_dense_kernel*
validate_shape(*
use_locking(*
T0
6
Const_4Const*
valueB:?*
dtype0
D
Const_5Const*%
valueB	"               *
dtype0	
e
StatelessRandomUniformStatelessRandomUniformConst_4Const_5*
T0*
Tseed0	*
dtype0
8
Const_6Const*
valueB 2Z/?4e??*
dtype0
?
Cast_1CastConst_6*

SrcT0*
Truncate( *

DstT0
G
Init_dense_2_dense_biasMulStatelessRandomUniformCast_1*
T0
?
Assign_dense_2_dense_biasAssigndense_2_dense_biasInit_dense_2_dense_bias*
use_locking(*
T0*
validate_shape(
i
dense_3_dense_kernel
VariableV2*
shared_name *
dtype0*
	container *
shape:	?d
b
dense_3_dense_bias
VariableV2*
dtype0*
	container *
shape:d*
shared_name 
<
Const_7Const*
valueB",  d   *
dtype0
D
Const_8Const*%
valueB	"               *
dtype0	
k
StatelessTruncatedNormal_1StatelessTruncatedNormalConst_7Const_8*
T0*
Tseed0	*
dtype0
8
Const_9Const*
valueB 2ar??B÷?*
dtype0
?
Cast_2CastConst_9*

SrcT0*
Truncate( *

DstT0
M
Init_dense_3_dense_kernelMulStatelessTruncatedNormal_1Cast_2*
T0
?
Assign_dense_3_dense_kernelAssigndense_3_dense_kernelInit_dense_3_dense_kernel*
use_locking(*
T0*
validate_shape(
6
Const_10Const*
valueB:d*
dtype0
E
Const_11Const*
dtype0	*%
valueB	"               
i
StatelessRandomUniform_1StatelessRandomUniformConst_10Const_11*
T0*
Tseed0	*
dtype0
9
Const_12Const*
valueB 2
c?Q??*
dtype0
@
Cast_3CastConst_12*

SrcT0*
Truncate( *

DstT0
I
Init_dense_3_dense_biasMulStatelessRandomUniform_1Cast_3*
T0
?
Assign_dense_3_dense_biasAssigndense_3_dense_biasInit_dense_3_dense_bias*
use_locking(*
T0*
validate_shape(
h
dense_4_dense_kernel
VariableV2*
dtype0*
	container *
shape
:d
*
shared_name 
b
dense_4_dense_bias
VariableV2*
dtype0*
	container *
shape:
*
shared_name 
=
Const_13Const*
valueB"d   
   *
dtype0
E
Const_14Const*%
valueB	"               *
dtype0	
m
StatelessTruncatedNormal_2StatelessTruncatedNormalConst_13Const_14*
T0*
Tseed0	*
dtype0
9
Const_15Const*
valueB 2???B???*
dtype0
@
Cast_4CastConst_15*

SrcT0*
Truncate( *

DstT0
M
Init_dense_4_dense_kernelMulStatelessTruncatedNormal_2Cast_4*
T0
?
Assign_dense_4_dense_kernelAssigndense_4_dense_kernelInit_dense_4_dense_kernel*
use_locking(*
T0*
validate_shape(
6
Const_16Const*
valueB:
*
dtype0
E
Const_17Const*%
valueB	"               *
dtype0	
i
StatelessRandomUniform_2StatelessRandomUniformConst_16Const_17*
T0*
Tseed0	*
dtype0
9
Const_18Const*
valueB 2Jh??|Z??*
dtype0
@
Cast_5CastConst_18*

SrcT0*
Truncate( *

DstT0
I
Init_dense_4_dense_biasMulStatelessRandomUniform_2Cast_5*
T0
?
Assign_dense_4_dense_biasAssigndense_4_dense_biasInit_dense_4_dense_bias*
use_locking(*
T0*
validate_shape(
6
PlaceholderPlaceholder*
shape:*
dtype0
7
numberOfLossesPlaceholder*
dtype0*
shape: 
J
ReshapeReshapedefault_data_placeholderConst*
T0*
Tshape0
^
MatMulMatMulReshapedense_2_dense_kernel*
transpose_a( *
transpose_b( *
T0
/
AddAddMatMuldense_2_dense_bias*
T0

ReluReluAdd*
T0
-
Activation_dense_2IdentityRelu*
T0
k
MatMul_1MatMulActivation_dense_2dense_3_dense_kernel*
T0*
transpose_a( *
transpose_b( 
3
Add_1AddMatMul_1dense_3_dense_bias*
T0

Relu_1ReluAdd_1*
T0
/
Activation_dense_3IdentityRelu_1*
T0
k
MatMul_2MatMulActivation_dense_3dense_4_dense_kernel*
transpose_a( *
transpose_b( *
T0
3
Add_2AddMatMul_2dense_4_dense_bias*
T0

Relu_2ReluAdd_2*
T0
/
Activation_dense_4IdentityRelu_2*
T0
h
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsActivation_dense_4Placeholder*
T0
2
Const_19Const*
value	B : *
dtype0
l
default_training_lossMeanSoftmaxCrossEntropyWithLogitsConst_19*

Tidx0*
	keep_dims( *
T0
>
Gradients/OnesLikeOnesLikedefault_training_loss*
T0
P
Gradients/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0
9
Gradients/ConstConst*
value	B : *
dtype0
;
Gradients/Const_1Const*
dtype0*
value	B :
@
Gradients/SizeSizeGradients/Shape*
T0*
out_type0
7
Gradients/AddAddConst_19Gradients/Size*
T0
<
Gradients/ModModGradients/AddGradients/Size*
T0
X
Gradients/RangeRangeGradients/ConstGradients/SizeGradients/Const_1*

Tidx0
8
Gradients/OnesLike_1OnesLikeGradients/Mod*
T0
?
Gradients/DynamicStitchDynamicStitchGradients/RangeGradients/ModGradients/ShapeGradients/OnesLike_1*
N*
T0
;
Gradients/Const_2Const*
dtype0*
value	B :
Q
Gradients/MaximumMaximumGradients/DynamicStitchGradients/Const_2*
T0
A
Gradients/DivDivGradients/ShapeGradients/Maximum*
T0
`
Gradients/ReshapeReshapeGradients/OnesLikeGradients/DynamicStitch*
T0*
Tshape0
S
Gradients/TileTileGradients/ReshapeGradients/Div*

Tmultiples0*
T0
R
Gradients/Shape_1ShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0
J
Gradients/Shape_2Shapedefault_training_loss*
T0*
out_type0
;
Gradients/Const_3Const*
value	B : *
dtype0
b
Gradients/ProdProdGradients/Shape_2Gradients/Const_3*

Tidx0*
	keep_dims( *
T0
d
Gradients/Prod_1ProdGradients/Shape_1Gradients/Const_3*

Tidx0*
	keep_dims( *
T0
;
Gradients/Const_4Const*
value	B :*
dtype0
J
Gradients/Maximum_1MaximumGradients/ProdGradients/Const_4*
T0
F
Gradients/Div_1DivGradients/Prod_1Gradients/Maximum_1*
T0
O
Gradients/CastCastGradients/Div_1*

SrcT0*
Truncate( *

DstT0
?
Gradients/Div_2DivGradients/TileGradients/Cast*
T0
J
Gradients/ZerosLike	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0
J
Gradients/Const_5/ConstConst*
valueB :
?????????*
dtype0
a
Gradients/ExpandDims
ExpandDimsGradients/Div_2Gradients/Const_5/Const*

Tdim0*
T0
Y
Gradients/MultiplyMulGradients/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0
?
Gradients/LogSoftmax
LogSoftmaxActivation_dense_4*
T0
D
Gradients/Const_6/ConstConst*
valueB
 *  ??*
dtype0
S
Gradients/Multiply_1MulGradients/LogSoftmaxGradients/Const_6/Const*
T0
J
Gradients/Const_7/ConstConst*
valueB :
?????????*
dtype0
c
Gradients/ExpandDims_1
ExpandDimsGradients/Div_2Gradients/Const_7/Const*

Tdim0*
T0
R
Gradients/Multiply_2MulGradients/ExpandDims_1Gradients/Multiply_1*
T0
;
Gradients/IdentityIdentityGradients/Multiply*
T0
B
Gradients/ReluGradReluGradGradients/IdentityAdd_2*
T0
=
Gradients/Identity_1IdentityGradients/ReluGrad*
T0
=
Gradients/Identity_2IdentityGradients/ReluGrad*
T0
=
Gradients/Shape_3ShapeMatMul_2*
T0*
out_type0
G
Gradients/Shape_4Shapedense_4_dense_bias*
T0*
out_type0
g
Gradients/BroadcastGradientArgsBroadcastGradientArgsGradients/Shape_3Gradients/Shape_4*
T0
q
Gradients/SumSumGradients/Identity_1Gradients/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
W
Gradients/Reshape_1ReshapeGradients/SumGradients/Shape_3*
T0*
Tshape0
u
Gradients/Sum_1SumGradients/Identity_2!Gradients/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_2ReshapeGradients/Sum_1Gradients/Shape_4*
T0*
Tshape0
t
Gradients/MatMulMatMulGradients/Reshape_1dense_4_dense_kernel*
T0*
transpose_a( *
transpose_b(
t
Gradients/MatMul_1MatMulActivation_dense_3Gradients/Reshape_1*
transpose_a(*
transpose_b( *
T0
;
Gradients/Identity_3IdentityGradients/MatMul*
T0
F
Gradients/ReluGrad_1ReluGradGradients/Identity_3Add_1*
T0
?
Gradients/Identity_4IdentityGradients/ReluGrad_1*
T0
?
Gradients/Identity_5IdentityGradients/ReluGrad_1*
T0
=
Gradients/Shape_5ShapeMatMul_1*
T0*
out_type0
G
Gradients/Shape_6Shapedense_3_dense_bias*
T0*
out_type0
i
!Gradients/BroadcastGradientArgs_1BroadcastGradientArgsGradients/Shape_5Gradients/Shape_6*
T0
u
Gradients/Sum_2SumGradients/Identity_4!Gradients/BroadcastGradientArgs_1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_3ReshapeGradients/Sum_2Gradients/Shape_5*
T0*
Tshape0
w
Gradients/Sum_3SumGradients/Identity_5#Gradients/BroadcastGradientArgs_1:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_4ReshapeGradients/Sum_3Gradients/Shape_6*
T0*
Tshape0
v
Gradients/MatMul_2MatMulGradients/Reshape_3dense_3_dense_kernel*
T0*
transpose_a( *
transpose_b(
t
Gradients/MatMul_3MatMulActivation_dense_2Gradients/Reshape_3*
transpose_a(*
transpose_b( *
T0
=
Gradients/Identity_6IdentityGradients/MatMul_2*
T0
D
Gradients/ReluGrad_2ReluGradGradients/Identity_6Add*
T0
?
Gradients/Identity_7IdentityGradients/ReluGrad_2*
T0
?
Gradients/Identity_8IdentityGradients/ReluGrad_2*
T0
;
Gradients/Shape_7ShapeMatMul*
out_type0*
T0
G
Gradients/Shape_8Shapedense_2_dense_bias*
T0*
out_type0
i
!Gradients/BroadcastGradientArgs_2BroadcastGradientArgsGradients/Shape_7Gradients/Shape_8*
T0
u
Gradients/Sum_4SumGradients/Identity_7!Gradients/BroadcastGradientArgs_2*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_5ReshapeGradients/Sum_4Gradients/Shape_7*
T0*
Tshape0
w
Gradients/Sum_5SumGradients/Identity_8#Gradients/BroadcastGradientArgs_2:1*

Tidx0*
	keep_dims( *
T0
Y
Gradients/Reshape_6ReshapeGradients/Sum_5Gradients/Shape_8*
T0*
Tshape0
v
Gradients/MatMul_4MatMulGradients/Reshape_5dense_2_dense_kernel*
transpose_b(*
T0*
transpose_a( 
i
Gradients/MatMul_5MatMulReshapeGradients/Reshape_5*
T0*
transpose_a(*
transpose_b( 
=
ShapeShapedense_2_dense_kernel*
T0*
out_type0
5
Const_20Const*
valueB
 *    *
dtype0
Y
%Init_optimizer_dense_2_dense_kernel-mFillShapeConst_20*
T0*

index_type0
v
 optimizer_dense_2_dense_kernel-m
VariableV2*
dtype0*
	container *
shape:
??*
shared_name 
?
'Assign_optimizer_dense_2_dense_kernel-mAssign optimizer_dense_2_dense_kernel-m%Init_optimizer_dense_2_dense_kernel-m*
use_locking(*
T0*
validate_shape(
?
Shape_1Shapedense_2_dense_kernel*
T0*
out_type0
5
Const_21Const*
valueB
 *    *
dtype0
[
%Init_optimizer_dense_2_dense_kernel-vFillShape_1Const_21*
T0*

index_type0
v
 optimizer_dense_2_dense_kernel-v
VariableV2*
shape:
??*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_dense_2_dense_kernel-vAssign optimizer_dense_2_dense_kernel-v%Init_optimizer_dense_2_dense_kernel-v*
use_locking(*
T0*
validate_shape(
=
Shape_2Shapedense_2_dense_bias*
T0*
out_type0
5
Const_22Const*
valueB
 *    *
dtype0
Y
#Init_optimizer_dense_2_dense_bias-mFillShape_2Const_22*
T0*

index_type0
o
optimizer_dense_2_dense_bias-m
VariableV2*
shared_name *
dtype0*
	container *
shape:?
?
%Assign_optimizer_dense_2_dense_bias-mAssignoptimizer_dense_2_dense_bias-m#Init_optimizer_dense_2_dense_bias-m*
use_locking(*
T0*
validate_shape(
=
Shape_3Shapedense_2_dense_bias*
out_type0*
T0
5
Const_23Const*
dtype0*
valueB
 *    
Y
#Init_optimizer_dense_2_dense_bias-vFillShape_3Const_23*
T0*

index_type0
o
optimizer_dense_2_dense_bias-v
VariableV2*
shared_name *
dtype0*
	container *
shape:?
?
%Assign_optimizer_dense_2_dense_bias-vAssignoptimizer_dense_2_dense_bias-v#Init_optimizer_dense_2_dense_bias-v*
use_locking(*
T0*
validate_shape(
?
Shape_4Shapedense_3_dense_kernel*
T0*
out_type0
5
Const_24Const*
valueB
 *    *
dtype0
[
%Init_optimizer_dense_3_dense_kernel-mFillShape_4Const_24*
T0*

index_type0
u
 optimizer_dense_3_dense_kernel-m
VariableV2*
dtype0*
	container *
shape:	?d*
shared_name 
?
'Assign_optimizer_dense_3_dense_kernel-mAssign optimizer_dense_3_dense_kernel-m%Init_optimizer_dense_3_dense_kernel-m*
use_locking(*
T0*
validate_shape(
?
Shape_5Shapedense_3_dense_kernel*
T0*
out_type0
5
Const_25Const*
valueB
 *    *
dtype0
[
%Init_optimizer_dense_3_dense_kernel-vFillShape_5Const_25*
T0*

index_type0
u
 optimizer_dense_3_dense_kernel-v
VariableV2*
shape:	?d*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_dense_3_dense_kernel-vAssign optimizer_dense_3_dense_kernel-v%Init_optimizer_dense_3_dense_kernel-v*
use_locking(*
T0*
validate_shape(
=
Shape_6Shapedense_3_dense_bias*
T0*
out_type0
5
Const_26Const*
valueB
 *    *
dtype0
Y
#Init_optimizer_dense_3_dense_bias-mFillShape_6Const_26*

index_type0*
T0
n
optimizer_dense_3_dense_bias-m
VariableV2*
shape:d*
shared_name *
dtype0*
	container 
?
%Assign_optimizer_dense_3_dense_bias-mAssignoptimizer_dense_3_dense_bias-m#Init_optimizer_dense_3_dense_bias-m*
use_locking(*
T0*
validate_shape(
=
Shape_7Shapedense_3_dense_bias*
T0*
out_type0
5
Const_27Const*
valueB
 *    *
dtype0
Y
#Init_optimizer_dense_3_dense_bias-vFillShape_7Const_27*
T0*

index_type0
n
optimizer_dense_3_dense_bias-v
VariableV2*
shared_name *
dtype0*
	container *
shape:d
?
%Assign_optimizer_dense_3_dense_bias-vAssignoptimizer_dense_3_dense_bias-v#Init_optimizer_dense_3_dense_bias-v*
validate_shape(*
use_locking(*
T0
?
Shape_8Shapedense_4_dense_kernel*
T0*
out_type0
5
Const_28Const*
valueB
 *    *
dtype0
[
%Init_optimizer_dense_4_dense_kernel-mFillShape_8Const_28*

index_type0*
T0
t
 optimizer_dense_4_dense_kernel-m
VariableV2*
shared_name *
dtype0*
	container *
shape
:d

?
'Assign_optimizer_dense_4_dense_kernel-mAssign optimizer_dense_4_dense_kernel-m%Init_optimizer_dense_4_dense_kernel-m*
validate_shape(*
use_locking(*
T0
?
Shape_9Shapedense_4_dense_kernel*
T0*
out_type0
5
Const_29Const*
valueB
 *    *
dtype0
[
%Init_optimizer_dense_4_dense_kernel-vFillShape_9Const_29*

index_type0*
T0
t
 optimizer_dense_4_dense_kernel-v
VariableV2*
shape
:d
*
shared_name *
dtype0*
	container 
?
'Assign_optimizer_dense_4_dense_kernel-vAssign optimizer_dense_4_dense_kernel-v%Init_optimizer_dense_4_dense_kernel-v*
use_locking(*
T0*
validate_shape(
>
Shape_10Shapedense_4_dense_bias*
T0*
out_type0
5
Const_30Const*
dtype0*
valueB
 *    
Z
#Init_optimizer_dense_4_dense_bias-mFillShape_10Const_30*
T0*

index_type0
n
optimizer_dense_4_dense_bias-m
VariableV2*
shared_name *
dtype0*
	container *
shape:

?
%Assign_optimizer_dense_4_dense_bias-mAssignoptimizer_dense_4_dense_bias-m#Init_optimizer_dense_4_dense_bias-m*
T0*
validate_shape(*
use_locking(
>
Shape_11Shapedense_4_dense_bias*
T0*
out_type0
5
Const_31Const*
valueB
 *    *
dtype0
Z
#Init_optimizer_dense_4_dense_bias-vFillShape_11Const_31*

index_type0*
T0
n
optimizer_dense_4_dense_bias-v
VariableV2*
dtype0*
	container *
shape:
*
shared_name 
?
%Assign_optimizer_dense_4_dense_bias-vAssignoptimizer_dense_4_dense_bias-v#Init_optimizer_dense_4_dense_bias-v*
use_locking(*
T0*
validate_shape(
a
optimizer_beta1_power
VariableV2*
dtype0*
	container *
shape: *
shared_name 
G
Init_optimizer_beta1_powerConst*
valueB
 *fff?*
dtype0
?
Assign_optimizer_beta1_powerAssignoptimizer_beta1_powerInit_optimizer_beta1_power*
T0*
validate_shape(*
use_locking(
a
optimizer_beta2_power
VariableV2*
shared_name *
dtype0*
	container *
shape: 
G
Init_optimizer_beta2_powerConst*
valueB
 *w??*
dtype0
?
Assign_optimizer_beta2_powerAssignoptimizer_beta2_powerInit_optimizer_beta2_power*
T0*
validate_shape(*
use_locking(
5
Const_32Const*
valueB
 *fff?*
dtype0
5
Const_33Const*
valueB
 *w??*
dtype0
5
Const_34Const*
valueB
 *o?:*
dtype0
5
Const_35Const*
valueB
 *???3*
dtype0
?
	ApplyAdam	ApplyAdamdense_2_dense_kernel optimizer_dense_2_dense_kernel-m optimizer_dense_2_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/MatMul_5*
T0*
use_nesterov( *
use_locking( 
?
ApplyAdam_1	ApplyAdamdense_2_dense_biasoptimizer_dense_2_dense_bias-moptimizer_dense_2_dense_bias-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/Reshape_6*
use_nesterov( *
use_locking( *
T0
?
ApplyAdam_2	ApplyAdamdense_3_dense_kernel optimizer_dense_3_dense_kernel-m optimizer_dense_3_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/MatMul_3*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_3	ApplyAdamdense_3_dense_biasoptimizer_dense_3_dense_bias-moptimizer_dense_3_dense_bias-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/Reshape_4*
use_locking( *
T0*
use_nesterov( 
?
ApplyAdam_4	ApplyAdamdense_4_dense_kernel optimizer_dense_4_dense_kernel-m optimizer_dense_4_dense_kernel-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/MatMul_1*
T0*
use_nesterov( *
use_locking( 
?
ApplyAdam_5	ApplyAdamdense_4_dense_biasoptimizer_dense_4_dense_bias-moptimizer_dense_4_dense_bias-voptimizer_beta1_poweroptimizer_beta2_powerConst_34Const_32Const_33Const_35Gradients/Reshape_2*
use_nesterov( *
use_locking( *
T0
4
MulMuloptimizer_beta1_powerConst_32*
T0
^
AssignAssignoptimizer_beta1_powerMul*
validate_shape(*
use_locking(*
T0
6
Mul_1Muloptimizer_beta2_powerConst_33*
T0
b
Assign_1Assignoptimizer_beta2_powerMul_1*
use_locking(*
T0*
validate_shape(
6
default_outputSoftmaxActivation_dense_4*
T0
2
Const_36Const*
value	B :*
dtype0
R
ArgMaxArgMaxdefault_outputConst_36*

Tidx0*
T0*
output_type0	
2
Const_37Const*
value	B :*
dtype0
Q
ArgMax_1ArgMaxPlaceholderConst_37*
T0*
output_type0	*

Tidx0
I
EqualEqualArgMaxArgMax_1*
incompatible_shape_error(*
T0	
=
Cast_6CastEqual*

SrcT0
*
Truncate( *

DstT0
2
Const_38Const*
value	B : *
dtype0
D
MeanMeanCast_6Const_38*
T0*

Tidx0*
	keep_dims( 
2
Const_39Const*
value	B :*
dtype0
T
ArgMax_2ArgMaxdefault_outputConst_39*
output_type0	*

Tidx0*
T0
2
Const_40Const*
dtype0*
value	B :
Q
ArgMax_3ArgMaxPlaceholderConst_40*

Tidx0*
T0*
output_type0	
M
Equal_1EqualArgMax_2ArgMax_3*
incompatible_shape_error(*
T0	
?
Cast_7CastEqual_1*

SrcT0
*
Truncate( *

DstT0
2
Const_41Const*
value	B : *
dtype0
F
Mean_1MeanCast_7Const_41*

Tidx0*
	keep_dims( *
T0 "?