"��
�DeviceCudnnRNNBackprop"(gradients/CudnnRNN_grad/CudnnRNNBackprop(�	1L7�A���@9�;c0;K@AL7�A���@I�;c0;K@Q���.o�?Y���.o�?�Unknown
aDeviceCudnnRNN"CudnnRNN(�	1�"�����@9ⶀ5�X2@A�"�����@Iⶀ5�X2@Q��~��}�?YPW�>�N�?�Unknown
DDeviceIDLE"IDLE1;�O����@A;�O����@Qׅ�R�?Y��_�y��?�Unknown
{Device	Transpose""gradients/transpose_grad/transpose(1;�O���@9'��e�t@A;�O���@I'��e�t@Q��!�?Y�@�yV�?�Unknown
dDevice	Transpose"transpose_0(1�l��)ٔ@9�l��)�t@A�l��)ٔ@I�l��)�t@Q���J��?Y�"�'��?�Unknown
dDevice	Transpose"transpose_9(1���Sc��@9!iJ���h@A���Sc��@I!iJ���h@Q��3�S�?Y6��w��?�Unknown
}Device	Transpose"$gradients/transpose_9_grad/transpose(1�ʡE���@9�c���h@A�ʡE���@I�c���h@Q^`8c1�?Y�yR;F�?�Unknown
�DeviceUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1����ב@9����ב@A����ב@I����ב@Q�ى?k�?YG�7P��?�Unknown
b	DeviceAddN"gradients/AddN(1��Q�Ƒ@9�:m��g@A��Q�Ƒ@I�:m��g@Q?�~<VZ�?Y��)�Q��?�Unknown
�
DeviceUnsortedSegmentSum"%Adam/Adam/update_9/UnsortedSegmentSum(1ףp=
��@9ףp=
��@Aףp=
��@Iףp=
��@Qי��y?YƉ]���?�Unknown
bDevice	ReverseV2"	ReverseV2(1���K�@9���Kv@A���K�@I���Kv@Qp�[��v?Y���:�?�Unknown
{Device	ReverseV2""gradients/ReverseV2_grad/ReverseV2(1D�l����@9D�l���u@AD�l����@ID�l���u@Q�bZ���v?YkE4�g�?�Unknown
bDevice	Transpose"	transpose(1���Q`�@9���Q`u@A���Q`�@I���Q`u@Q�q���v?YO�,���?�Unknown
gDeviceAddN"Adam/gradients/AddN(1^�I��@9^�I��@A^�I��@I^�I��@Q"G&8�q?Y�>�#��?�Unknown
�DeviceStridedSliceGrad"-gradients/strided_slice_grad/StridedSliceGrad(1����Gx@9!iJ��/P@A����Gx@I!iJ��/P@Q��wgi?Y���4��?�Unknown
tDeviceConcatV2"model_4/bidirectional/concat(1Zd;�O�w@9Zd;�O�g@AZd;�O�w@IZd;�O�g@Q�?y2��h?Y	OG���?�Unknown
xDevice	ReverseV2"model_4/bidirectional/ReverseV2(1ףp=
r@9ףp=
r@Aףp=
r@Iףp=
r@Qd]�Ǣc?Yf�����?�Unknown
�Device	ReverseV2"-gradient_tape/model_4/bidirectional/ReverseV2(1-���nr@9-���nr@A-���nr@I-���nr@Q�Q�k�c?Y�bzh��?�Unknown
oDeviceUnique"Adam/Adam/update_9/Unique(
1+��Jn@9Έ���;8@A+��Jn@IΈ���;8@QE39�fF_?YRǛ��?�Unknown
mDeviceUnique"Adam/Adam/update/Unique(
1�z�G5i@9��v��*4@A�z�G5i@I��v��*4@QJÜZ?Y]$)�+�?�Unknown
~DeviceSlice")gradient_tape/model_4/bidirectional/Slice(1�G�zh@9�G�zh@A�G�zh@I�G�zh@Q�c�P�X?Y�i8�?�Unknown
�DeviceSlice"+gradient_tape/model_4/bidirectional/Slice_1(1fffffh@9fffffh@Afffffh@Ifffffh@Q�Ae�*�X?Y���'xD�?�Unknown
bDeviceConcatV2"
concat_1_0(1��x�&�d@9Sq��3�+@A��x�&�d@ISq��3�+@QTа|G�U?Y9�K;O�?�Unknown
�DeviceResourceGather"$model_4/token_embed/embedding_lookup(1-���nb@9-���nb@A-���nb@I-���nb@Q�Q�k�S?Y���'�X�?�Unknown
iDeviceAddN"Adam/gradients/AddN_1(1m����"[@9m����"[@Am����"[@Im����"[@Q-�s6JL?Y��K:�_�?�Unknown
�Device_Send"?model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor/_65(1;�O��W@9;�O��W@A;�O��W@I;�O��W@Q�n�R�G?Y��εe�?�Unknown
uDeviceConcatV2"gradients/split_2_grad/concat(01V-ZV@9!�rh���?AV-ZV@I!�rh���?Quf���G?Y^���zk�?�Unknown
ZDeviceSplit"split(1�Zd;/S@9;&x0O�)@A�Zd;/S@I;&x0O�)@Q�����C?Y�>Odnp�?�Unknown
�DeviceResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1j�t��R@9j�t��R@Aj�t��R@Ij�t��R@Q�=�D'GC?Yb .@u�?�Unknown
�Device_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_11(1�����tQ@9�����tQ@A�����tQ@I�����tQ@Q]yp�B?Yk����y�?�Unknown
sDeviceConcatV2"gradients/split_grad/concat(1�C�l�P@9Zd;�/@A�C�l�P@IZd;�/@Q,l[.A?Ym[S$~�?�Unknown
� DeviceResourceGather"#model_4/char_embed/embedding_lookup(1�C�l�P@9�C�l�P@A�C�l�P@I�C�l�P@Q,l[.A?Yo6�X��?�Unknown
�!DeviceSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1)\���(L@9)\���(L@A)\���(L@I)\���(L@QqW�!�=?Y�o����?�Unknown
\"DeviceSplit"split_1(1�x�&1L@9㥛� �"@A�x�&1L@I㥛� �"@Q�?�i��<?YB��"���?�Unknown
u#DeviceConcatV2"gradients/split_1_grad/concat(1P��n#K@9���a@AP��n#K@I���a@QD�+l<?Y�]���?�Unknown
r$Device	ZerosLike"Adam/gradients/zeros_like(1���K�F@9���K�F@A���K�F@I���K�F@QH(V`B7?Y��g���?�Unknown
t%Device	ZerosLike"Adam/gradients/zeros_like_3(1���K�F@9���K�F@A���K�F@I���K�F@QH(V`B7?YȘrH��?�Unknown
i&DeviceMul"Adam/Adam/update/mul_3(1��x�&�E@9��x�&�E@A��x�&�E@I��x�&�E@Q����36?Y�[ſ���?�Unknown
}'Device	Transpose"$gradients/transpose_1_grad/transpose(1/�$�E@9y�&1�@A/�$�E@Iy�&1�@Q��_��36?Y���2w��?�Unknown
�(DeviceResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1-����C@9-����C@A-����C@I-����C@Q�n����4?YxG�
��?�Unknown
�)Device_Send"Gmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2/_23(1V-�A@9V-�A@AV-�A@IV-�A@QhH!_ E2?Y�+(�S��?�Unknown
u*Device	ZerosLike"Adam/gradients/zeros_like_12(1�� �rhA@9�� �rhA@A�� �rhA@I�� �rhA@Q&���1?Y3O`����?�Unknown
�+Device_Send"2model_4/char_vectorizer/string_lookup/SelectV2/_57(1ףp=
'A@9ףp=
'A@Aףp=
'A@Iףp=
'A@Q�e�%z�1?Y@	�^ɡ�?�Unknown
d,Device	Transpose"transpose_1(1j�t��>@9U�}�{@Aj�t��>@IU�}�{@Q5f+��/?Y����ģ�?�Unknown
i-DeviceMul"Adam/Adam/update/mul_2(1�O��n�=@9�O��n�=@A�O��n�=@I�O��n�=@Q�wUk
�.?Y�tNj���?�Unknown
g.DeviceMul"Adam/Adam/update/mul(1V-�=@9V-�=@AV-�=@IV-�=@Q}s��ƨ.?YQ������?�Unknown
^/DeviceConcatV2"concat(1%��C�<@9nY�c�@A%��C�<@InY�c�@Qt��ILx-?Y{0{q��?�Unknown
d0Device	Transpose"transpose_4(1���S�;@9;�O��n@A���S�;@I;�O��n@Q���A{�,?Y�L33:��?�Unknown
�1DeviceResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1�~j�t�8@9�~j�t�8@A�~j�t�8@I�~j�t�8@Q����_)?Y�e�+Ь�?�Unknown
�2DeviceResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1�~j�t�8@9�~j�t�8@A�~j�t�8@I�~j�t�8@Q����_)?Y�~s$f��?�Unknown
d3Device	Transpose"transpose_5(1Zd;�O�7@9ͅ�)g@AZd;�O�7@Iͅ�)g@QQ�A�P(?Y��2��?�Unknown
\4DeviceSplit"split_2(11�Zd7@9��:&x0@A1�Zd7@I��:&x0@Q�e~[�&(?Y�N]�m��?�Unknown
}5Device	Transpose"$gradients/transpose_5_grad/transpose(1��C�l�6@9W��	@A��C�l�6@IW��	@Q$J�*�B'?Y�����?�Unknown
d6Device	Transpose"transpose_2(1�I+�6@9��e��	@A�I+�6@I��e��	@Q�Ep�>B'?Y�h�U��?�Unknown
d7Device	Transpose"transpose_3(1-���f6@9��|?5�@A-���f6@I��|?5�@Qx2�r� '?Y�A?�ǵ�?�Unknown
}8Device	Transpose"$gradients/transpose_3_grad/transpose(1�z�G�5@9,�Œ_�@A�z�G�5@I,�Œ_�@Qǌ�j�3&?Y��1+��?�Unknown
�9DeviceResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1/�$�5@9/�$�5@A/�$�5@I/�$�5@Q��_��3&?Y��j���?�Unknown
}:Device	Transpose"$gradients/transpose_4_grad/transpose(1/�$�5@9y�&1�@A/�$�5@Iy�&1�@Q��_��3&?Y����?�Unknown
};Device	Transpose"$gradients/transpose_7_grad/transpose(1B`��"{4@9+��N@AB`��"{4@I+��N@QkϾ�6%%?Y�ƈ�C��?�Unknown
d<Device	Transpose"transpose_8(1�|?5^Z4@9=Q�F(#@A�|?5^Z4@I=Q�F(#@Q����a%?Yx�-���?�Unknown
}=Device	Transpose"$gradients/transpose_8_grad/transpose(1�E���t3@9�\H�R�	@A�E���t3@I�\H�R�	@Q��$?YY���ս�?�Unknown
}>Device	Transpose"$gradients/transpose_2_grad/transpose(1��~j�t3@9$M�8��	@A��~j�t3@I$M�8��	@Q�>AM$?Y:�����?�Unknown
d?Device	Transpose"transpose_6(1y�&1L3@9��ޖ��	@Ay�&1L3@I��ޖ��	@Qxt�G�#?Y����U��?�Unknown
d@Device	Transpose"transpose_7(1����K3@9�|?5^�	@A����K3@I�|?5^�	@QHp;[�#?Y��0����?�Unknown
�ADevice_Send"Bmodel_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor/_67(1��"��~2@9��"��~2@A��"��~2@I��"��~2@QW\.ݑ#?Y��N	���?�Unknown
�BDevice	_HostRecv"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_30(1�|?5^�1@9�|?5^�1@A�|?5^�1@I�|?5^�1@QS�!��M"?Y�T�����?�Unknown
}CDevice	Transpose"$gradients/transpose_6_grad/transpose(1�� �rh1@9а+@�5@A�� �rh1@Iа+@�5@Q&���!?Ytf�r
��?�Unknown
�DDevice_Send"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_45(1X9��1@9X9��1@AX9��1@IX9��1@Q'\����!?Y�Nݽ$��?�Unknown
lEDeviceMatMul"model_4/dense_7/MatMul(19��v��*@99��v��*@A9��v��*@I9��v��*@Qf;���|?Y�\�� ��?�Unknown
�FDevice_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_17(1?5^�I*@9?5^�I*@A?5^�I*@I?5^�I*@Q�ЩE��?YӉK����?�Unknown
�GDeviceResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1�~j�t�(@9�~j�t�(@A�~j�t�(@I�~j�t�(@Q����_?Ya��Ţ��?�Unknown
sHDeviceSoftmax"model_4/output_layer/Softmax(1�~j�t�(@9�~j�t�(@A�~j�t�(@I�~j�t�(@Q����_?Y���m��?�Unknown
jIDeviceMatMul"model_4/dense/MatMul(1H�z��&@9H�z��&@AH�z��&@IH�z��&@QSNP��B?YqE�'��?�Unknown
|JDeviceMatMul"&gradient_tape/model_4/dense_4/MatMul_1(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q�Ep�>B?Y�P����?�Unknown
�KDeviceMatMul"+gradient_tape/model_4/output_layer/MatMul_1(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q�Ep�>B?Yu\����?�Unknown
qLDeviceMatMul"model_4/output_layer/MatMul(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q�Ep�>B?Y�g�V��?�Unknown
zMDeviceMatMul"$gradient_tape/model_4/dense/MatMul_1(1{�G�z$@9{�G�z$@A{�G�z$@I{�G�z$@Q;�N�$?Ymr�5���?�Unknown
zNDeviceMatMul"$gradient_tape/model_4/dense_8/MatMul(1{�G�z$@9{�G�z$@A{�G�z$@I{�G�z$@Q;�N�$?Y�|&]���?�Unknown
zODeviceMatMul"$gradient_tape/model_4/dense_4/MatMul(1�v��o"@9�v��o"@A�v��o"@I�v��o"@Q�X�.?YN��@��?�Unknown
xPDeviceMatMul""gradient_tape/model_4/dense/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y�&�����?�Unknown
|QDeviceMatMul"&gradient_tape/model_4/dense_1/MatMul_1(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y$0q��?�Unknown
zRDeviceMatMul"$gradient_tape/model_4/dense_7/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y�9QV	��?�Unknown
|SDeviceMatMul"&gradient_tape/model_4/dense_8/MatMul_1(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y�B�����?�Unknown
TDeviceMatMul")gradient_tape/model_4/output_layer/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?YeL��9��?�Unknown
lUDeviceMatMul"model_4/dense_3/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y�U���?�Unknown
lVDeviceMatMul"model_4/dense_5/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q�P-��?Y;_AKj��?�Unknown
zWDeviceMatMul"$gradient_tape/model_4/dense_3/MatMul(1�A`��b @9�A`��b @A�A`��b @I�A`��b @Q'��S��?Y��[����?�Unknown
zXDeviceMatMul"$gradient_tape/model_4/dense_1/MatMul(1����Mb @9����Mb @A����Mb @I����Mb @Q��\�?Y�<�x��?�Unknown
|YDeviceMatMul"&gradient_tape/model_4/dense_7/MatMul_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\�?YXH ��?�Unknown
�ZDeviceCumsum"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum(1����Mb @9����Mb @A����Mb @I����Mb @Q��\�?Y������?�Unknown
l[DeviceMatMul"model_4/dense_8/MatMul(1����Mb @9����Mb @A����Mb @I����Mb @Q��\�?Y ����?�Unknown
�\DeviceCumsum"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum(1����Mb @9����Mb @A����Mb @I����Mb @Q��\�?Yu(�@���?�Unknown
|]DeviceMatMul"&gradient_tape/model_4/dense_2/MatMul_1(1�O��nR @9�O��nR @A�O��nR @I�O��nR @QR�����?Y����?�Unknown
�^DeviceBiasAddGrad"1gradient_tape/model_4/dense_1/BiasAdd/BiasAddGrad(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q ��!�?Y�y���?�Unknown
z_DeviceMatMul"$gradient_tape/model_4/dense_5/MatMul(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q ��!�?Yr���	��?�Unknown
|`DeviceMatMul"&gradient_tape/model_4/dense_6/MatMul_1(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q ��!�?Y��J���?�Unknown
�aDevice_Send"Lmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2/_27(1)\���(@9)\���(@A)\���(@I)\���(@QqW�!�?Y��H����?�Unknown
�bDevice_Send"7model_4/text_vectorization/string_lookup_1/SelectV2/_61(1��"��~@9��"��~@A��"��~@I��"��~@Q�#�� [?Y��Kb��?�Unknown
�cDeviceResourceApplyAdam"%Adam/Adam/update_32/ResourceApplyAdam(1�G�z�@9�G�z�@A�G�z�@I�G�z�@Qj�Q��`	?YI�����?�Unknown
�dDeviceResourceApplyAdam"%Adam/Adam/update_16/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?YIO�-��?�Unknown
\eDeviceArgMax"ArgMax(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?Y�U����?�Unknown
^fDeviceArgMax"ArgMax_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?Y�[&����?�Unknown
�gDeviceBiasAddGrad"/gradient_tape/model_4/dense/BiasAdd/BiasAddGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?YbN|]��?�Unknown
�hDeviceBiasAddGrad"6gradient_tape/model_4/output_layer/BiasAdd/BiasAddGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?Yehv����?�Unknown
liDeviceMatMul"model_4/dense_1/MatMul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?Y�n�x(��?�Unknown
�jDeviceRandomUniform"4model_4/dropout/dropout/random_uniform/RandomUniform(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_	?Y�t�����?�Unknown
�kDeviceConcatV2".model_4/token_char_positional_embedding/concat(1�~j�t�@9����Mb @A�~j�t�@I����Mb @Q����_	?Y:{�t���?�Unknown
�lDevice_Send"/model_4/char_vectorizer/string_lookup/Equal/_21(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@Q�p]	�?Y����K��?�Unknown
�mDeviceResourceApplyAdam"%Adam/Adam/update_10/ResourceApplyAdam(1�C�l�{@9�C�l�{@A�C�l�{@I�C�l�{@Q���&?Y�<T���?�Unknown
�nDeviceAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y&B�����?�Unknown
�oDeviceResourceApplyAdam"%Adam/Adam/update_11/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?YaG�{I��?�Unknown
�pDeviceResourceApplyAdam"%Adam/Adam/update_12/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�Li���?�Unknown
�qDeviceResourceApplyAdam"%Adam/Adam/update_20/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�Q5����?�Unknown
�rDeviceResourceApplyAdam"%Adam/Adam/update_22/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?YW7G��?�Unknown
�sDeviceResourceApplyAdam"%Adam/Adam/update_23/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?YM\�ʛ��?�Unknown
�tDeviceResourceApplyAdam"%Adam/Adam/update_26/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�a�^���?�Unknown
�uDeviceResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�fe�D��?�Unknown
�vDeviceBiasAddGrad"1gradient_tape/model_4/dense_3/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�k1����?�Unknown
�wDeviceBiasAddGrad"1gradient_tape/model_4/dense_5/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y9q����?�Unknown
�xDeviceBiasAddGrad"1gradient_tape/model_4/dense_8/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?YtvɭB��?�Unknown
�yDeviceStridedSlice"3model_4/char_vectorizer/StringSplit/strided_slice_1(1{�G�z@9{�G�z@A{�G�z@I{�G�z@Q;�N�$?Y�{�A���?�Unknown
�zDeviceBiasAddGrad"1gradient_tape/model_4/dense_4/BiasAdd/BiasAddGrad(1�MbX9@9�MbX9@A�MbX9@I�MbX9@Q�J�I�?Y�������?�Unknown
�{DeviceResourceApplyAdam"%Adam/Adam/update_13/ResourceApplyAdam(1j�t�@9j�t�@Aj�t�@Ij�t�@Q8�H�t�?Y�%��=��?�Unknown
l|DeviceMatMul"model_4/dense_4/MatMul(1���Sc@9���Sc@A���Sc@I���Sc@Q��˦j� ?Y+�9r���?�Unknown
�}DeviceResourceApplyAdam"%Adam/Adam/update_18/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?YZũ���?�Unknown
�~DeviceResourceApplyAdam"%Adam/Adam/update_24/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�����?�Unknown
�DeviceResourceApplyAdam"%Adam/Adam/update_30/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�͉nL��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_34/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y������?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_36/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�i����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_38/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?YE��j��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_39/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Yt�I[��?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�⹽���?�Unknown
��DeviceAssignSubVariableOp"&Adam/Adam/update_9/AssignSubVariableOp(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y��)g���?�Unknown
��DeviceTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�&��?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_2/BiasAdd/BiasAddGrad(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y0�	�i��?�Unknown
{�DeviceMatMul"$gradient_tape/model_4/dense_2/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y_�yc���?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_6/BiasAdd/BiasAddGrad(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y������?�Unknown
w�DeviceConcatV2"model_4/bidirectional_1/concat(1����Mb@9����Mb @A����Mb@I����Mb @Q��\� ?Y��Y�4��?�Unknown
��DeviceCast"Zmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y���_x��?�Unknown
��DeviceConcatV2"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat(1����Mb@9����Mb @A����Mb@I����Mb @Q��\� ?Y:	���?�Unknown
��DeviceSelectV2".model_4/char_vectorizer/string_lookup/SelectV2(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?YJ�����?�Unknown
m�DeviceMatMul"model_4/dense_2/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Yy\C��?�Unknown
o�DeviceBiasAdd"model_4/dense_6/BiasAdd(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y�����?�Unknown
m�DeviceMatMul"model_4/dense_6/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?Y������?�Unknown
��DeviceGreaterEqual"$model_4/dropout/dropout/GreaterEqual(1����Mb@9����Mb@A����Mb@I����Mb@Q��\� ?YjX��?�Unknown
��DeviceConcatV2"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat(1����Mb@9����Mb @A����Mb@I����Mb @Q��\� ?Y5�R��?�Unknown
��DeviceConcatV2"*model_4/token_char_hybrid_embedding/concat(1����Mb@9����Mb @A����Mb@I����Mb @Q��\� ?Yd!J����?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_7/BiasAdd/BiasAddGrad(1T㥛� @9T㥛� @AT㥛� @IT㥛� @Qr���� ?Y�eF���?�Unknown
��Device	_HostRecv"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_32(1�E����@9�E����@A�E����@I�E����@Q�q3��>Y�H{���?�Unknown
��Device_Send"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_49(1�E����@9�E����@A�E����@I�E����@Q�q3��>Y�+��>��?�Unknown
��DeviceAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1/�$��@9/�$��@A/�$��@I/�$��@Q(�M�a�>Y��0�q��?�Unknown
~�DeviceSum"*categorical_crossentropy/weighted_loss/Sum(1/�$��@9/�$��@A/�$��@I/�$��@Q(�M�a�>Y#`Y���?�Unknown
t�DeviceBiasAdd"model_4/output_layer/BiasAdd(1/�$��@9/�$��@A/�$��@I/�$��@Q(�M�a�>YG�����?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Yk���	��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_14/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y� ��<��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_15/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�
Zo��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_17/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_19/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�	2����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_21/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YF���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_25/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YCZV:��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_27/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Ygnm��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_28/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y��ԟ��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_29/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y������?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y��R��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_33/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y��8��?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_8/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y#��j��?�Unknown
��DeviceResourceScatterAdd"%Adam/Adam/update_9/ResourceScatterAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y?&揝��?�Unknown
��DeviceResourceScatterAdd"'Adam/Adam/update_9/ResourceScatterAdd_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Yc)�N���?�Unknown
l�DeviceSqrt"Adam/Adam/update_9/Sqrt(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�,��?�Unknown
l�DeviceAddV2"Adam/Adam/update_9/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�/"�5��?�Unknown
j�DeviceMul"Adam/Adam/update_9/mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�26�h��?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_3(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�5JK���?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_4(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y9^
���?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_5(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y;<r� ��?�Unknown
r�DeviceRealDiv"Adam/Adam/update_9/truediv(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y_?��3��?�Unknown
\�DevicePow"Adam/Pow(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�B�Gf��?�Unknown
^�DeviceAddV2"Adam/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�E����?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_11(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�H�����?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_17(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�Kք���?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_9(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YO�C1��?�Unknown
h�Device_Recv"IteratorGetNext/_4(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y7R�d��?�Unknown
��DeviceDivNoNan",categorical_crossentropy/weighted_loss/value(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y[U��?�Unknown
��DeviceMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YX&����?�Unknown
��DeviceDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�[:@���?�Unknown
}�DeviceReluGrad"$gradient_tape/model_4/dense/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�^N�.��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_2/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�ab�a��?�Unknown
{�DeviceMatMul"$gradient_tape/model_4/dense_6/MatMul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Yev}���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_7/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y3h�<���?�Unknown
�DeviceMul"+gradient_tape/model_4/dropout/dropout/Mul_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YWk�����?�Unknown
a�DeviceCast"model_4/Cast(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y{n��,��?�Unknown
�DeviceFill"*model_4/bidirectional/forward_lstm_2/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�q�y_��?�Unknown
��DeviceFill",model_4/bidirectional_1/forward_lstm_3/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�t�8���?�Unknown
m�DeviceBiasAdd"model_4/dense/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y�w�����?�Unknown
o�DeviceBiasAdd"model_4/dense_5/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y{����?�Unknown
o�DeviceBiasAdd"model_4/dense_7/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y/~v*��?�Unknown
o�DeviceMul"model_4/dropout/dropout/Mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>YS�*5]��?�Unknown
q�DeviceMul"model_4/dropout/dropout/Mul_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Yw�>���?�Unknown
g�DeviceFill"model_4/lstm/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y��R����?�Unknown
i�DeviceFill"model_4/lstm_1/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q����_�>Y��fr���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_37/ResourceApplyAdam(1R���Q@9R���Q@AR���Q@IR���Q@QV�����>Yڭ'�'��?�Unknown
W�DeviceMul"Mul(1R���Q@9R���Q@AR���Q@IR���Q@QV�����>Y����Y��?�Unknown
��DeviceSlice"9gradient_tape/model_4/token_char_hybrid_embedding/Slice_1(1R���Q@9R���Q@AR���Q@IR���Q@QV�����>Y�����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_35/ResourceApplyAdam(1���Mb@9���Mb@A���Mb@I���Mb@Q�a�!7��>Y#7ʽ��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_2(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@QQ�A�P�>Y%��k���?�Unknown
`�DeviceCast"Adam/Cast_1(11�Zd @91�Zd @A1�Zd @I1�Zd @QC��Ly��>Y=S�D��?�Unknown
��DeviceAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YUU�2��?�Unknown
��DeviceAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YmWC�S��?�Unknown
~�DeviceReadVariableOp"Adam/Adam/update/ReadVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�Y��u��?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�[�����?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�]kl���?�Unknown
j�DeviceSqrt"Adam/Adam/update/Sqrt(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�_#A���?�Unknown
j�DeviceAddV2"Adam/Adam/update/add(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�a����?�Unknown
j�DeviceMul"Adam/Adam/update/mul_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�c����?�Unknown
j�DeviceMul"Adam/Adam/update/mul_4(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YfK�@��?�Unknown
j�DeviceMul"Adam/Adam/update/mul_5(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y-h�b��?�Unknown
p�DeviceRealDiv"Adam/Adam/update/truediv(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YEj�h���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_31/ResourceApplyAdam(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y]ls=���?�Unknown
��DeviceAssignVariableOp"#Adam/Adam/update_9/AssignVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yun+���?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update_9/ReadVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�p�����?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�r����?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�tS�-��?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�veO��?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�x�9q��?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y{{���?�Unknown
^�DeviceSqrt"	Adam/Sqrt(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y}3���?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y5����?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_10(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YM������?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_13(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Ye�[a��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_14(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y}�6<��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_15(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���
^��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_16(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y������?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_4(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yŋ;����?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_5(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yݍ����?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_6(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���]���?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_7(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�c2��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_8(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y%�)��?�Unknown
\�DeviceMul"Adam/mul(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y=���J��?�Unknown
^�DeviceSub"
Adam/sub_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YU���l��?�Unknown
^�DeviceSub"
Adam/sub_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Ym�C����?�Unknown
^�DeviceSub"
Adam/sub_3(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���Y���?�Unknown
d�DeviceRealDiv"Adam/truediv(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���.���?�Unknown
w�DeviceAssignAddVariableOp"AssignAddVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��k���?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y͢#���?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�۬7��?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_4(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y����Y��?�Unknown
Y�DeviceCast"Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�KV{��?�Unknown
[�DeviceCast"Cast_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y-�+���?�Unknown
[�DeviceCast"Cast_3(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YE������?�Unknown
[�DeviceEqual"Equal(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y]�s����?�Unknown
h�Device_Send"IteratorGetNext/_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yu�+���?�Unknown
h�Device_Recv"IteratorGetNext/_6(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���}$��?�Unknown
h�Device
LogicalAnd"
LogicalAnd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���RF��?�Unknown�
Y�DeviceSum"Sum_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��S'h��?�Unknown
c�DeviceDivNoNan"
div_no_nan(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yչ����?�Unknown
e�DeviceDivNoNan"div_no_nan_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��Ы��?�Unknown
��DeviceSlice"+gradient_tape/model_4/bidirectional_1/Slice(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�{����?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_1/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�3z���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_3/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y5��N��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_4/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YMģ#3��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_5/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Ye�[�T��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_6/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y}��v��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_8/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��ˡ���?�Unknown
}�DeviceMul")gradient_tape/model_4/dropout/dropout/Mul(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�̃v���?�Unknown
��DeviceSlice";gradient_tape/model_4/token_char_positional_embedding/Slice(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��;K���?�Unknown
��DeviceSlice"=gradient_tape/model_4/token_char_positional_embedding/Slice_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y������?�Unknown
��DeviceSlice"=gradient_tape/model_4/token_char_positional_embedding/Slice_2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�ҫ���?�Unknown
��DeviceCast"cmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�c�A��?�Unknown
��DeviceStridedSlice"1model_4/char_vectorizer/StringSplit/strided_slice(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y%��c��?�Unknown
g�DeviceRelu"model_4/dense/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y=��r���?�Unknown
o�DeviceBiasAdd"model_4/dense_1/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YUۋG���?�Unknown
i�DeviceRelu"model_4/dense_1/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Ym�C���?�Unknown
o�DeviceBiasAdd"model_4/dense_2/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�������?�Unknown
i�DeviceRelu"model_4/dense_2/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�����?�Unknown
o�DeviceBiasAdd"model_4/dense_3/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��k�.��?�Unknown
i�DeviceRelu"model_4/dense_3/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��#oP��?�Unknown
o�DeviceBiasAdd"model_4/dense_4/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���Cr��?�Unknown
i�DeviceRelu"model_4/dense_4/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�����?�Unknown
i�DeviceCast"model_4/dense_5/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y�K���?�Unknown
i�DeviceRelu"model_4/dense_5/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y-�����?�Unknown
i�DeviceRelu"model_4/dense_6/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>YE����?�Unknown
i�DeviceRelu"model_4/dense_7/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y]�sk��?�Unknown
o�DeviceBiasAdd"model_4/dense_8/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Yu�+@=��?�Unknown
i�DeviceRelu"model_4/dense_8/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���_��?�Unknown
q�DeviceCast"model_4/dropout/dropout/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y������?�Unknown
��DeviceCast"]model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y��S����?�Unknown
��DeviceCast"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y������?�Unknown
��DeviceCast"fmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y���g���?�Unknown
��DeviceStridedSlice"4model_4/text_vectorization/StringSplit/strided_slice(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y|<��?�Unknown
��DeviceStridedSlice"6model_4/text_vectorization/StringSplit/strided_slice_1(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y4*��?�Unknown
��DeviceSelectV2"3model_4/text_vectorization/string_lookup_1/SelectV2(1����Mb @9����Mb @A����Mb @I����Mb @Q��\��>Y5��K��?�Unknown
^�DevicePow"
Adam/Pow_1(1T㥛�  @9T㥛�  @AT㥛�  @IT㥛�  @Qr�����>YD'Q3m��?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_1(1T㥛�  @9T㥛�  @AT㥛�  @IT㥛�  @Qr�����>YSI�����?�Unknown
��DeviceSlice"-gradient_tape/model_4/bidirectional_1/Slice_1(1X9��v��?9X9��v��?AX9��v��?IX9��v��?Qw!	c�>YZ��F���?�Unknown
��Device_Send"kmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_53(1�������?9�������?A�������?I�������?Q
~��/n�>Y�L�����?�Unknown
��Device_Send"Zmodel_4_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value/_15(1+�����?9+�����?A+�����?I+�����?Q�lF!���>YBn�R���?�Unknown
��Device_Recv"7categorical_crossentropy/weighted_loss/num_elements/_74(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?Q�>AM�>Y���h���?�Unknown
��Device_Recv"jmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/_34(1�l�����?9�l�����?A�l�����?I�l�����?Q0�5a���>Y�����?�Unknown
��DeviceAssignVariableOp"%Adam/Adam/update_9/AssignVariableOp_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y�<���?�Unknown
\�DeviceSub"Adam/sub(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y���'��?�Unknown
[�DeviceCast"Cast_4(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y���8��?�Unknown
��DeviceCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y�P�I��?�Unknown
��DeviceSlice"7gradient_tape/model_4/token_char_hybrid_embedding/Slice(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y���Z��?�Unknown
��DeviceCast"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Q��\��>Y�vk��?�Unknown
��Device_Send"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_59(1X9��v��?9X9��v��?AX9��v��?IX9��v��?Qw!	c�>Y8�{��?�Unknown
^�Device_Recv"Size/_76(1���Q��?9���Q��?A���Q��?I���Q��?Q�0��l��>Y�xǴ���?�Unknown
��Device_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_19(1���Q��?9���Q��?A���Q��?I���Q��?Q�0��l��>Y��}����?�Unknown
��Device_Send"nmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_55(1V-��?9V-��?AV-��?IV-��?Q}s��ƨ�>Y�����?�Unknown
��Device	_HostRecv"gmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast/_38(1�������?9�������?A�������?I�������?Q
~��/n�>Y������?�Unknown
��Device	_HostRecv"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1/_42(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q����_�>Y�������?�Unknown
��Device	_HostRecv"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1/_44(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q����_�>YM��{���?�Unknown
f�Device_Recv"strided_slice/_8(1Zd;�O��?9Zd;�O��?AZd;�O��?IZd;�O��?QQ�A�P�>Y�����?�Unknown
��Device_Send"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_63(1�I+��?9�I+��?A�I+��?I�I+��?Q�Ep�>B�>Y�^E���?�Unknown
��Device_Send"Tmodel_4_char_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value/_9(1�I+��?9�I+��?A�I+��?I�I+��?Q�Ep�>B�>Y~�3����?�Unknown
��Device	_HostRecv"jmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast/_40(1/�$��?9/�$��?A/�$��?I/�$��?Q��_��3�>Y     �?�Unknown
^�Device_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_13(Y     �?�Unknown
��Device_Recv"mmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/_36(Y     �?�Unknown
]�Device_Send"4model_4/text_vectorization/string_lookup_1/Equal/_25(Y     �?�Unknown
C�HostIDLE"IDLE1H�z�5�@AH�z�5�@af��16�?if��16�?�Unknown
��HostStringSplitV2"1model_4/char_vectorizer/StringSplit/StringSplitV2(1�K7�A֑@9�K7�A֑@A�K7�A֑@I�K7�A֑@a�g��ŝ?i]$�?�Unknown
��HostStaticRegexReplace"*model_4/char_vectorizer/StaticRegexReplace(1;�O����@9;�O����@A;�O����@I;�O����@aq���?ikT���?�Unknown
��HostStaticRegexReplace"-model_4/text_vectorization/StaticRegexReplace(1�Q����@9�Q����@A�Q����@I�Q����@aѺ��Yӓ?i�p)�Op�?�Unknown
��HostLookupTableFindV2"Cmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2(1����M"�@9����M"�@A����M"�@I����M"�@aF�i&�͐?i�\'���?�Unknown
w�HostDataset"!Iterator::Root::Prefetch::BatchV2(1��(\s�@9��(\s�@A7�A`�˃@I7�A`�˃@aZ��vm��?iO���z�?�Unknown
��HostStringSplitV2"4model_4/text_vectorization/StringSplit/StringSplitV2(1�$��B�@9�$��B�@A�$��B�@I�$��B�@a�qq�ό?iQ��#��?�Unknown
}�HostDataset"&Iterator::Root::Prefetch::BatchV2::Zip(�1'1���@9'1��@Ah��|?�~@Ih��|?��?a8�+��ȉ?i� ��FU�?�Unknown
��HostDataset"6Iterator::Root::Prefetch::BatchV2::Zip[0]::TensorSlice(�1���Ʃ}@9���Ʃ�?A���Ʃ}@I���Ʃ�?a8�h��?i#�JL��?�Unknown
��HostLookupTableFindV2"Hmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2(1�Q���w@9�Q���w@A�Q���w@I�Q���w@a,˘�L��?iPD}�?�Unknown
��HostDataset"6Iterator::Root::Prefetch::BatchV2::Zip[1]::TensorSlice(�1��C�l�u@9��C�l��?A��C�l�u@I��C�l��?ah[?���?i�A��OO�?�Unknown
��HostRaggedTensorToTensor";model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor(1� �rh�l@9� �rh�l@A� �rh�l@I� �rh�l@a�l���w?i�bq�A�?�Unknown
�HostEqual"+model_4/char_vectorizer/string_lookup/Equal(1�A`�Жf@9�A`�Жf@A�A`�Жf@I�A`�Жf@a�M��r?i��U����?�Unknown
}�HostStringLower"#model_4/char_vectorizer/StringLower(1���Sse@9���Sse@A���Sse@I���Sse@a�����q?i��J���?�Unknown
��HostBincount"gmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount(1�O��n2e@9�O��n2e@A�O��n2e@I�O��n2e@a����q?i�K,�$��?�Unknown
f�Host_Send"IteratorGetNext/_1(1�"��~
_@9�"��~
_@A�"��~
_@I�"��~
_@a߈ѐ��i?i-�Z�?�Unknown
��HostStringLower"&model_4/text_vectorization/StringLower(1�G�z,]@9�G�z,]@A�G�z,]@I�G�z,]@a5�.��Xh?iLO2e�?�Unknown
��HostEqual"0model_4/text_vectorization/string_lookup_1/Equal(1^�IV@9^�IV@A^�IV@I^�IV@a��l�^b?i�ջ��0�?�Unknown
��HostRaggedTensorToTensor">model_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor(1�Zd[U@9�Zd[U@A�Zd[U@I�Zd[U@a�����a?i�v���B�?�Unknown
��Host_Send"Gmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2/_23(1V-�S@9V-�S@AV-�S@IV-�S@a(����N`?i�7lQ�R�?�Unknown
��Host_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_17(1H�z�gN@9H�z�gN@AH�z�gN@IH�z�gN@a�j���_Y?iB��C�_�?�Unknown
��Host_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_19(1��� �rM@9��� �rM@A��� �rM@I��� �rM@aB�2o�X?ic�X��k�?�Unknown
��Host_Send"4model_4/text_vectorization/string_lookup_1/Equal/_25(1H�z�GL@9H�z�GL@AH�z�GL@IH�z�GL@ah��w�W?i���w�?�Unknown
��Host_Send"Lmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2/_27(1bX9�K@9bX9�K@AbX9�K@IbX9�K@a������V?i������?�Unknown
��Host_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_13(1�����H@9�����H@A�����H@I�����H@a�P]��T?i;�jD5��?�Unknown
��Host_Send"kmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_53(1�S㥛�G@9�S㥛�G@A�S㥛�G@I�S㥛�G@a��&��S?i��}���?�Unknown
��Host_Send"Bmodel_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor/_67(1�����=G@9�����=G@A�����=G@I�����=G@aP��vqeS?i�X9?���?�Unknown
��Host_Send"/model_4/char_vectorizer/string_lookup/Equal/_21(1�O��nrF@9�O��nrF@A�O��nrF@I�O��nrF@a�����R?i���?�Unknown
f�Host
LogicalAnd"
LogicalAnd(1=
ףp�E@9=
ףp�E@A=
ףp�E@I=
ףp�E@a�50\?R?i6�&�5��?�Unknown�
��HostBincount"jmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount(19��v��D@99��v��D@A9��v��D@I9��v��D@aI���(Q?i:�6ʻ�?�Unknown
��Host_Send"?model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor/_65(1X9���C@9X9���C@AX9���C@IX9���C@a�O)�P?i���
��?�Unknown
��Host_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_11(1�����LC@9�����LC@A�����LC@I�����LC@aq��[P?i�Čy��?�Unknown
v�HostFlushSummaryWriter"FlushSummaryWriter(1�I+�B@9�I+�B@A�I+�B@I�I+�B@a+����<O?i��m����?�Unknown�
��Host_Send"nmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_55(1���Mb @@9���Mb @@A���Mb @@I���Mb @@a#�̵�J?i5��d���?�Unknown
j�HostWriteSummary"WriteSummary(1���S�e7@9���S�e7@A���S�e7@I���S�e7@aڕ���C?i�B���?�Unknown�
��Host_Recv"7model_4/text_vectorization/string_lookup_1/SelectV2/_62(1!�rh�M6@9!�rh�M6@A!�rh�M6@I!�rh�M6@a�3��B?i��Y+��?�Unknown
d�HostDataset"Iterator::Root(1X9��vND@9X9��vND@A��|?5�5@I��|?5�5@a����ZB?i�O���?�Unknown
��Host_Recv"2model_4/char_vectorizer/string_lookup/SelectV2/_58(1��"���2@9��"���2@A��"���2@I��"���2@a�ӑ��I??i��HF���?�Unknown
n�HostDataset"Iterator::Root::Prefetch(1��Q��2@9��Q��2@A��Q��2@I��Q��2@aJ]�)??i��kˍ��?�Unknown
��Host	_HostSend";gradient_tape/model_4/token_embed/embedding_lookup/Size/_71(1��� �r'@9��� �r'@A��� �r'@I��� �r'@aQ�X���3?i�_����?�Unknown
��Host_Recv"jmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum/_48(1���S�!@9���S�!@A���S�!@I���S�!@a�L*b�-?i������?�Unknown
��Host	_HostSend":gradient_tape/model_4/char_embed/embedding_lookup/Size/_69(1���S�@9���S�@A���S�@I���S�@a�^��)�(?i �i��?�Unknown
��Host_Recv"Umodel_4_char_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value/_10(1ˡE���@9ˡE���@AˡE���@IˡE���@alz��]\'?i4
�[���?�Unknown
��Host_Recv"mmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum/_52(1������@9������@A������@I������@a��=6�^&?in�DE��?�Unknown
f�Host_Send"IteratorGetNext/_3(1u�V@9u�V@Au�V@Iu�V@aڠæ��%?iJ��0���?�Unknown
m�HostIteratorGetNext"IteratorGetNext(1�C�l��@9�C�l��@A�C�l��@I�C�l��@a �6$?iy;r���?�Unknown
��Host_Recv"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_46(1�V�@9�V�@A�V�@I�V�@a����UA!?iV�l����?�Unknown
��Host_Recv"Zmodel_4_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value/_16(1V-�@9V-�@AV-�@IV-�@a��m�?i"�w���?�Unknown
��Host_Recv"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_64(1�G�z@9�G�z@A�G�z@I�G�z@awh���?iboD���?�Unknown
��Host_Recv"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_60(1w��/�@9w��/�@Aw��/�@Iw��/�@aH�U;�i?iJ���?�Unknown
��Host_Recv"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_50(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?a���W/
?i)�eM}��?�Unknown
f�Host_Send"IteratorGetNext/_5(1ffffff�?9ffffff�?Affffff�?Iffffff�?a3�L��?i[�	���?�Unknown
b�HostIdentity"Identity(1��n���?9��n���?A��n���?I��n���?aN�S���>i     �?�Unknown�*��
�DeviceCudnnRNNBackprop"(gradients/CudnnRNN_grad/CudnnRNNBackprop(�	1L7�A���@9�;c0;K@AL7�A���@I�;c0;K@Q��NO�?Y��NO�?�Unknown
aDeviceCudnnRNN"CudnnRNN(�	1�"�����@9ⶀ5�X2@A�"�����@Iⶀ5�X2@Q^���]�?Y��i��&�?�Unknown
{Device	Transpose""gradients/transpose_grad/transpose(1;�O���@9'��e�t@A;�O���@I'��e�t@QsMA�铒?Y�\��?�Unknown
dDevice	Transpose"transpose_0(1�l��)ٔ@9�l��)�t@A�l��)ٔ@I�l��)�t@Q��D��?YV��~��?�Unknown
dDevice	Transpose"transpose_9(1���Sc��@9!iJ���h@A���Sc��@I!iJ���h@Q
2�烆?Y?4�y�?�Unknown
}Device	Transpose"$gradients/transpose_9_grad/transpose(1�ʡE���@9�c���h@A�ʡE���@I�c���h@Qib�bf[�?Y����%��?�Unknown
�DeviceUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1����ב@9����ב@A����ב@I����ב@Q(�t�?Ya��{�(�?�Unknown
bDeviceAddN"gradients/AddN(1��Q�Ƒ@9�:m��g@A��Q�Ƒ@I�:m��g@QrM�d>a�?Y�=Cu~~�?�Unknown
�	DeviceUnsortedSegmentSum"%Adam/Adam/update_9/UnsortedSegmentSum(1ףp=
��@9ףp=
��@Aףp=
��@Iףp=
��@QA58�~?Y�y����?�Unknown
b
Device	ReverseV2"	ReverseV2(1���K�@9���Kv@A���K�@I���Kv@Q���!�z?Y#��Ӹ��?�Unknown
{Device	ReverseV2""gradients/ReverseV2_grad/ReverseV2(1D�l����@9D�l���u@AD�l����@ID�l���u@Qc��pz?Y���$�?�Unknown
bDevice	Transpose"	transpose(1���Q`�@9���Q`u@A���Q`�@I���Q`u@QN����y?Y���X�?�Unknown
gDeviceAddN"Adam/gradients/AddN(1^�I��@9^�I��@A^�I��@I^�I��@QG��ot?Y6)[���?�Unknown
�DeviceStridedSliceGrad"-gradients/strided_slice_grad/StridedSliceGrad(1����Gx@9!iJ��/P@A����Gx@I!iJ��/P@Q��a,�3m?Yߊ�z��?�Unknown
tDeviceConcatV2"model_4/bidirectional/concat(1Zd;�O�w@9Zd;�O�g@AZd;�O�w@IZd;�O�g@Q�W�3�l?Y7}��Ϻ�?�Unknown
xDevice	ReverseV2"model_4/bidirectional/ReverseV2(1ףp=
r@9ףp=
r@Aףp=
r@Iףp=
r@Qap��>f?YQ�x���?�Unknown
�Device	ReverseV2"-gradient_tape/model_4/bidirectional/ReverseV2(1-���nr@9-���nr@A-���nr@I-���nr@Q�FAI9+f?Y�.��9��?�Unknown
oDeviceUnique"Adam/Adam/update_9/Unique(
1+��Jn@9Έ���;8@A+��Jn@IΈ���;8@Q�80�t7b?Y�^�Iq��?�Unknown
mDeviceUnique"Adam/Adam/update/Unique(
1�z�G5i@9��v��*4@A�z�G5i@I��v��*4@Q�v��}Q^?Y�Y���?�Unknown
~DeviceSlice")gradient_tape/model_4/bidirectional/Slice(1�G�zh@9�G�zh@A�G�zh@I�G�zh@QX���U�\?Y[+~��?�Unknown
�DeviceSlice"+gradient_tape/model_4/bidirectional/Slice_1(1fffffh@9fffffh@Afffffh@Ifffffh@Qz,�N��\?Yq�%�%�?�Unknown
bDeviceConcatV2"
concat_1_0(1��x�&�d@9Sq��3�+@A��x�&�d@ISq��3�+@Q�h�Y?Y�Y$�2�?�Unknown
�DeviceResourceGather"$model_4/token_embed/embedding_lookup(1-���nb@9-���nb@A-���nb@I-���nb@Q�FAI9+V?Y���G)=�?�Unknown
iDeviceAddN"Adam/gradients/AddN_1(1m����"[@9m����"[@Am����"[@Im����"[@Q5rZ��QP?Y˧�RE�?�Unknown
�Device_Send"?model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor/_65(1;�O��W@9;�O��W@A;�O��W@I;�O��W@Qr����K?Y��ICL�?�Unknown
uDeviceConcatV2"gradients/split_2_grad/concat(01V-ZV@9!�rh���?AV-ZV@I!�rh���?Q�i5�-�J?Y �{��R�?�Unknown
ZDeviceSplit"split(1�Zd;/S@9;&x0O�)@A�Zd;/S@I;&x0O�)@QR�:��G?Y˷+��X�?�Unknown
�DeviceResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1j�t��R@9j�t��R@Aj�t��R@Ij�t��R@Qb�0quF?Y��]^�?�Unknown
�Device_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_11(1�����tQ@9�����tQ@A�����tQ@I�����tQ@Q@+����D?Y�1��c�?�Unknown
sDeviceConcatV2"gradients/split_grad/concat(1�C�l�P@9Zd;�/@A�C�l�P@IZd;�/@Q� "0yD?YV:�]�h�?�Unknown
�DeviceResourceGather"#model_4/char_embed/embedding_lookup(1�C�l�P@9�C�l�P@A�C�l�P@I�C�l�P@Q� "0yD?Y�BA<�m�?�Unknown
� DeviceSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1)\���(L@9)\���(L@A)\���(L@I)\���(L@Q}�=|.�@?YVR��q�?�Unknown
\!DeviceSplit"split_1(1�x�&1L@9㥛� �"@A�x�&1L@I㥛� �"@Q��y�@?Y�Z�v�?�Unknown
u"DeviceConcatV2"gradients/split_1_grad/concat(1P��n#K@9���a@AP��n#K@I���a@Q	��K�Q@?Y���R&z�?�Unknown
r#Device	ZerosLike"Adam/gradients/zeros_like(1���K�F@9���K�F@A���K�F@I���K�F@Qp^��q;?Y���`�}�?�Unknown
t$Device	ZerosLike"Adam/gradients/zeros_like_3(1���K�F@9���K�F@A���K�F@I���K�F@Qp^��q;?Y��o��?�Unknown
i%DeviceMul"Adam/Adam/update/mul_3(1��x�&�E@9��x�&�E@A��x�&�E@I��x�&�E@Q8$��(�9?Y^�2(��?�Unknown
}&Device	Transpose"$gradients/transpose_1_grad/transpose(1/�$�E@9y�&1�@A/�$�E@Iy�&1�@Q��u-�9?YvX�c��?�Unknown
�'DeviceResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1-����C@9-����C@A-����C@I-����C@Q�L8�:8?Y!��;d��?�Unknown
�(Device_Send"Gmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2/_23(1V-�A@9V-�A@AV-�A@IV-�A@Q���H5?Y���L��?�Unknown
u)Device	ZerosLike"Adam/gradients/zeros_like_12(1�� �rhA@9�� �rhA@A�� �rhA@I�� �rhA@Q���a��4?Y��\H���?�Unknown
�*Device_Send"2model_4/char_vectorizer/string_lookup/SelectV2/_57(1ףp=
'A@9ףp=
'A@Aףp=
'A@Iףp=
'A@Q��+~1�4?Y.��n?��?�Unknown
d+Device	Transpose"transpose_1(1j�t��>@9U�}�{@Aj�t��>@IU�}�{@Q��eqy2?Y�^�����?�Unknown
i,DeviceMul"Adam/Adam/update/mul_2(1�O��n�=@9�O��n�=@A�O��n�=@I�O��n�=@Q�)����1?Y��Rʖ�?�Unknown
g-DeviceMul"Adam/Adam/update/mul(1V-�=@9V-�=@AV-�=@IV-�=@Q'�ob��1?Y�����?�Unknown
^.DeviceConcatV2"concat(1%��C�<@9nY�c�@A%��C�<@InY�c�@Qh[�5L*1?Y�؅�*��?�Unknown
d/Device	Transpose"transpose_4(1���S�;@9;�O��n@A���S�;@I;�O��n@Q/\�0?Y#��>��?�Unknown
�0DeviceResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1�~j�t�8@9�~j�t�8@A�~j�t�8@I�~j�t�8@Q7u=�܎-?Y�,����?�Unknown
�1DeviceResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1�~j�t�8@9�~j�t�8@A�~j�t�8@I�~j�t�8@Q7u=�܎-?Yѐ����?�Unknown
d2Device	Transpose"transpose_5(1Zd;�O�7@9ͅ�)g@AZd;�O�7@Iͅ�)g@Q�:P��S,?Y�������?�Unknown
\3DeviceSplit"split_2(11�Zd7@9��:&x0@A1�Zd7@I��:&x0@Q�)[#P",?Y���x��?�Unknown
}4Device	Transpose"$gradients/transpose_5_grad/transpose(1��C�l�6@9W��	@A��C�l�6@IW��	@Q��2�+?Y�&s�)��?�Unknown
d5Device	Transpose"transpose_2(1�I+�6@9��e��	@A�I+�6@I��e��	@Q� c`J+?Y�,(ۧ�?�Unknown
d6Device	Transpose"transpose_3(1-���f6@9��|?5�@A-���f6@I��|?5�@Q�R0�*?Y��;���?�Unknown
}7Device	Transpose"$gradients/transpose_3_grad/transpose(1�z�G�5@9,�Œ_�@A�z�G�5@I,�Œ_�@Q߁��O�)?Y��(��?�Unknown
�8DeviceResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1/�$�5@9/�$�5@A/�$�5@I/�$�5@Q��u-�)?YY�,�Ŭ�?�Unknown
}9Device	Transpose"$gradients/transpose_4_grad/transpose(1/�$�5@9y�&1�@A/�$�5@Iy�&1�@Q��u-�)?Y��?�c��?�Unknown
}:Device	Transpose"$gradients/transpose_7_grad/transpose(1B`��"{4@9+��N@AB`��"{4@I+��N@Q�G���(?Y	j����?�Unknown
d;Device	Transpose"transpose_8(1�|?5^Z4@9=Q�F(#@A�|?5^Z4@I=Q�F(#@Qa�w��z(?Y�цzu��?�Unknown
}<Device	Transpose"$gradients/transpose_8_grad/transpose(1�E���t3@9�\H�R�	@A�E���t3@I�\H�R�	@Qp虽f'?Yp`���?�Unknown
}=Device	Transpose"$gradients/transpose_2_grad/transpose(1��~j�t3@9$M�8��	@A��~j�t3@I$M�8��	@Q!R��nf'?Y��LMb��?�Unknown
d>Device	Transpose"transpose_6(1y�&1L3@9��ޖ��	@Ay�&1L3@I��ޖ��	@QW��)z5'?Y��յ�?�Unknown
d?Device	Transpose"transpose_7(1����K3@9�|?5^�	@A����K3@I�|?5^�	@QA�W+5'?YM��H��?�Unknown
�@Device_Send"Bmodel_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor/_67(1��"��~2@9��"��~2@A��"��~2@I��"��~2@Q���'�>&?Y�G嬸�?�Unknown
�ADevice	_HostRecv"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_30(1�|?5^�1@9�|?5^�1@A�|?5^�1@I�|?5^�1@Q���AcR%?Y̛{��?�Unknown
}BDevice	Transpose"$gradients/transpose_6_grad/transpose(1�� �rh1@9а+@�5@A�� �rh1@Iа+@�5@Q���a��$?YڷA	Q��?�Unknown
�CDevice_Send"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_45(1X9��1@9X9��1@AX9��1@IX9��1@Q�v#T��$?Y��㙼�?�Unknown
lDDeviceMatMul"model_4/dense_7/MatMul(19��v��*@99��v��*@A9��v��*@I9��v��*@Q���� ?Y�Z ���?�Unknown
�EDevice_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_17(1?5^�I*@9?5^�I*@A?5^�I*@I?5^�I*@Q�xr�T?YdV�����?�Unknown
�FDeviceResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1�~j�t�(@9�~j�t�(@A�~j�t�(@I�~j�t�(@Q7u=�܎?YP��&���?�Unknown
sGDeviceSoftmax"model_4/output_layer/Softmax(1�~j�t�(@9�~j�t�(@A�~j�t�(@I�~j�t�(@Q7u=�܎?Y<�|�m��?�Unknown
jHDeviceMatMul"model_4/dense/MatMul(1H�z��&@9H�z��&@AH�z��&@IH�z��&@Qew��?Y �dF��?�Unknown
|IDeviceMatMul"&gradient_tape/model_4/dense_4/MatMul_1(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q� c`J?Y8�'��?�Unknown
�JDeviceMatMul"+gradient_tape/model_4/output_layer/MatMul_1(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q� c`J?YP�b����?�Unknown
qKDeviceMatMul"model_4/output_layer/MatMul(1�I+�&@9�I+�&@A�I+�&@I�I+�&@Q� c`J?Yh뵫���?�Unknown
zLDeviceMatMul"$gradient_tape/model_4/dense/MatMul_1(1{�G�z$@9{�G�z$@A{�G�z$@I{�G�z$@QY�����?Y��u����?�Unknown
zMDeviceMatMul"$gradient_tape/model_4/dense_8/MatMul(1{�G�z$@9{�G�z$@A{�G�z$@I{�G�z$@QY�����?Y�5�Z��?�Unknown
zNDeviceMatMul"$gradient_tape/model_4/dense_4/MatMul(1�v��o"@9�v��o"@A�v��o"@I�v��o"@Q��G9�+?Y,^O%��?�Unknown
xODeviceMatMul""gradient_tape/model_4/dense/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?Y�|~���?�Unknown
|PDeviceMatMul"&gradient_tape/model_4/dense_1/MatMul_1(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?Y���n��?�Unknown
zQDeviceMatMul"$gradient_tape/model_4/dense_7/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?YN�0 ��?�Unknown
|RDeviceMatMul"&gradient_tape/model_4/dense_8/MatMul_1(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?Y������?�Unknown
SDeviceMatMul")gradient_tape/model_4/output_layer/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?Ya�.���?�Unknown
lTDeviceMatMul"model_4/dense_3/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?Y�>[<4��?�Unknown
lUDeviceMatMul"model_4/dense_5/MatMul(1;�O��n"@9;�O��n"@A;�O��n"@I;�O��n"@Q���%+?YC䇕���?�Unknown
zVDeviceMatMul"$gradient_tape/model_4/dense_3/MatMul(1�A`��b @9�A`��b @A�A`��b @I�A`��b @Qm�0�?Y�?���?�Unknown
zWDeviceMatMul"$gradient_tape/model_4/dense_1/MatMul(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.��?YI��� ��?�Unknown
|XDeviceMatMul"&gradient_tape/model_4/dense_7/MatMul_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.��?Y�lA����?�Unknown
�YDeviceCumsum"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.��?Y���,\��?�Unknown
lZDeviceMatMul"model_4/dense_8/MatMul(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.��?Y Zt����?�Unknown
�[DeviceCumsum"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.��?Y��v���?�Unknown
|\DeviceMatMul"&gradient_tape/model_4/dense_2/MatMul_1(1�O��nR @9�O��nR @A�O��nR @I�O��nR @QtF>@|�?Y���4��?�Unknown
�]DeviceBiasAddGrad"1gradient_tape/model_4/dense_1/BiasAdd/BiasAddGrad(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q/�� >?Yx�q���?�Unknown
z^DeviceMatMul"$gradient_tape/model_4/dense_5/MatMul(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q/�� >?YAb�aH��?�Unknown
|_DeviceMatMul"&gradient_tape/model_4/dense_6/MatMul_1(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@Q/�� >?Y
�R���?�Unknown
�`Device_Send"Lmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2/_27(1)\���(@9)\���(@A)\���(@I)\���(@Q}�=|.�?Y��v�Y��?�Unknown
�aDevice_Send"7model_4/text_vectorization/string_lookup_1/SelectV2/_61(1��"��~@9��"��~@A��"��~@I��"��~@Q_B��?Y⢎C���?�Unknown
�bDeviceResourceApplyAdam"%Adam/Adam/update_32/ResourceApplyAdam(1�G�z�@9�G�z�@A�G�z�@I�G�z�@Qqbp�?Y���O��?�Unknown
�cDeviceResourceApplyAdam"%Adam/Adam/update_16/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Y��a����?�Unknown
\dDeviceArgMax"ArgMax(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Y���;��?�Unknown
^eDeviceArgMax"ArgMax_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Y�+H6���?�Unknown
�fDeviceBiasAddGrad"/gradient_tape/model_4/dense/BiasAdd/BiasAddGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Y|D�q(��?�Unknown
�gDeviceBiasAddGrad"6gradient_tape/model_4/output_layer/BiasAdd/BiasAddGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Yr].����?�Unknown
lhDeviceMatMul"model_4/dense_1/MatMul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Yhv����?�Unknown
�iDeviceRandomUniform"4model_4/dropout/dropout/random_uniform/RandomUniform(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎?Y^�$���?�Unknown
�jDeviceConcatV2".model_4/token_char_positional_embedding/concat(1�~j�t�@9����Mb @A�~j�t�@I����Mb @Q7u=�܎?YT��_��?�Unknown
�kDevice_Send"/model_4/char_vectorizer/string_lookup/Equal/_21(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@QJ��	?Y���5h��?�Unknown
�lDeviceResourceApplyAdam"%Adam/Adam/update_10/ResourceApplyAdam(1�C�l�{@9�C�l�{@A�C�l�{@I�C�l�{@Q�y�C�?Y�Ӵ����?�Unknown
�mDeviceAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?YĽ�H-��?�Unknown
�nDeviceResourceApplyAdam"%Adam/Adam/update_11/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y�tϏ��?�Unknown
�oDeviceResourceApplyAdam"%Adam/Adam/update_12/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y�TV���?�Unknown
�pDeviceResourceApplyAdam"%Adam/Adam/update_20/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y*|4�T��?�Unknown
�qDeviceResourceApplyAdam"%Adam/Adam/update_22/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?YLfd���?�Unknown
�rDeviceResourceApplyAdam"%Adam/Adam/update_23/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?YnP����?�Unknown
�sDeviceResourceApplyAdam"%Adam/Adam/update_26/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y�:�q|��?�Unknown
�tDeviceResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y�$�����?�Unknown
�uDeviceBiasAddGrad"1gradient_tape/model_4/dense_3/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y��A��?�Unknown
�vDeviceBiasAddGrad"1gradient_tape/model_4/dense_5/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y��s���?�Unknown
�wDeviceBiasAddGrad"1gradient_tape/model_4/dense_8/BiasAdd/BiasAddGrad(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y�S���?�Unknown
�xDeviceStridedSlice"3model_4/char_vectorizer/StringSplit/strided_slice_1(1{�G�z@9{�G�z@A{�G�z@I{�G�z@QY�����?Y:�3i��?�Unknown
�yDeviceBiasAddGrad"1gradient_tape/model_4/dense_4/BiasAdd/BiasAddGrad(1�MbX9@9�MbX9@A�MbX9@I�MbX9@Q�=ͭ�R?Yo��_���?�Unknown
�zDeviceResourceApplyAdam"%Adam/Adam/update_13/ResourceApplyAdam(1j�t�@9j�t�@Aj�t�@Ij�t�@Q��o�|+?Y-��+��?�Unknown
l{DeviceMatMul"model_4/dense_4/MatMul(1���Sc@9���Sc@A���Sc@I���Sc@Q��xε?YG���y��?�Unknown
�|DeviceResourceApplyAdam"%Adam/Adam/update_18/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�=C����?�Unknown
�}DeviceResourceApplyAdam"%Adam/Adam/update_24/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y������?�Unknown
�~DeviceResourceApplyAdam"%Adam/Adam/update_30/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y4��[f��?�Unknown
�DeviceResourceApplyAdam"%Adam/Adam/update_34/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�o).���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_36/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�*v ��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_38/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y!���R��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_39/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Yp�����?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�\\w���?�Unknown
��DeviceAssignSubVariableOp"&Adam/Adam/update_9/AssignSubVariableOp(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�I?��?�Unknown
��DeviceTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y]�����?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_2/BiasAdd/BiasAddGrad(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y��B����?�Unknown
{�DeviceMatMul"$gradient_tape/model_4/dense_2/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�I��+��?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_6/BiasAdd/BiasAddGrad(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?YJܒz��?�Unknown
w�DeviceConcatV2"model_4/bidirectional_1/concat(1����Mb@9����Mb @A����Mb@I����Mb @Qz��.��?Y��(e���?�Unknown
��DeviceCast"Zmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y�{u7��?�Unknown
��DeviceConcatV2"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat(1����Mb@9����Mb @A����Mb@I����Mb @Qz��.��?Y77�	g��?�Unknown
��DeviceSelectV2".model_4/char_vectorizer/string_lookup/SelectV2(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y��ܵ��?�Unknown
m�DeviceMatMul"model_4/dense_2/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Yխ[���?�Unknown
o�DeviceBiasAdd"model_4/dense_6/BiasAdd(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y$i��S��?�Unknown
m�DeviceMatMul"model_4/dense_6/MatMul(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Ys$�R���?�Unknown
��DeviceGreaterEqual"$model_4/dropout/dropout/GreaterEqual(1����Mb@9����Mb@A����Mb@I����Mb@Qz��.��?Y��A%���?�Unknown
��DeviceConcatV2"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat(1����Mb@9����Mb @A����Mb@I����Mb @Qz��.��?Y���?��?�Unknown
��DeviceConcatV2"*model_4/token_char_hybrid_embedding/concat(1����Mb@9����Mb @A����Mb@I����Mb @Qz��.��?Y`V�Ɏ��?�Unknown
��DeviceBiasAddGrad"1gradient_tape/model_4/dense_7/BiasAdd/BiasAddGrad(1T㥛� @9T㥛� @AT㥛� @IT㥛� @Q�T��e?Y���`���?�Unknown
��Device	_HostRecv"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_32(1�E����@9�E����@A�E����@I�E����@Q������>Y�=��?�Unknown
��Device_Send"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_49(1�E����@9�E����@A�E����@I�E����@Q������>Y�*��S��?�Unknown
��DeviceAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1/�$��@9/�$��@A/�$��@I/�$��@Q�O�XS��>Y��A����?�Unknown
~�DeviceSum"*categorical_crossentropy/weighted_loss/Sum(1/�$��@9/�$��@A/�$��@I/�$��@Q�O�XS��>Y3�����?�Unknown
t�DeviceBiasAdd"model_4/output_layer/BiasAdd(1/�$��@9/�$��@A/�$��@I/�$��@Q�O�XS��>Yz>�?��?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��H]@��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_14/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YpW{{��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_15/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�㻘���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_17/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Yfpu����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_19/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��.�,��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_21/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y\���g��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_25/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_27/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YR�[-���?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_28/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�.K��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_29/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YH��hT��?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�G�����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_33/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y>�A����?�Unknown
��DeviceResourceApplyAdam"$Adam/Adam/update_8/ResourceApplyAdam(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�`����?�Unknown
��DeviceResourceScatterAdd"%Adam/Adam/update_9/ResourceScatterAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y4��@��?�Unknown
��DeviceResourceScatterAdd"'Adam/Adam/update_9/ResourceScatterAdd_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�yn�{��?�Unknown
l�DeviceSqrt"Adam/Adam/update_9/Sqrt(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y*(���?�Unknown
l�DeviceAddV2"Adam/Adam/update_9/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y���8���?�Unknown
j�DeviceMul"Adam/Adam/update_9/mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y �V-��?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_3(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��Tth��?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_4(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y8����?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_5(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��ǯ���?�Unknown
r�DeviceRealDiv"Adam/Adam/update_9/truediv(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YQ����?�Unknown
\�DevicePow"Adam/Pow(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��:�T��?�Unknown
^�DeviceAddV2"Adam/add(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Yj����?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_11(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y}��&���?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_17(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��gD��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_9(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Ys!bA��?�Unknown
h�Device_Recv"IteratorGetNext/_4(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��|��?�Unknown
��DeviceDivNoNan",categorical_crossentropy/weighted_loss/value(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Yi(�����?�Unknown
��DeviceMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�M����?�Unknown
��DeviceDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y_A�-��?�Unknown
}�DeviceReluGrad"$gradient_tape/model_4/dense/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y����h��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_2/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YUZz���?�Unknown
{�DeviceMatMul"$gradient_tape/model_4/dense_6/MatMul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y��32���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_7/ReluGrad(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YKs�O��?�Unknown
�DeviceMul"+gradient_tape/model_4/dropout/dropout/Mul_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y���mU��?�Unknown
a�DeviceCast"model_4/Cast(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>YA�`����?�Unknown
�DeviceFill"*model_4/bidirectional/forward_lstm_2/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�����?�Unknown
��DeviceFill",model_4/bidirectional_1/forward_lstm_3/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y7�����?�Unknown
m�DeviceBiasAdd"model_4/dense/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�1��A��?�Unknown
o�DeviceBiasAdd"model_4/dense_5/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y-�F}��?�Unknown
o�DeviceBiasAdd"model_4/dense_7/BiasAdd(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�J  ���?�Unknown
o�DeviceMul"model_4/dropout/dropout/Mul(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y#׹=���?�Unknown
q�DeviceMul"model_4/dropout/dropout/Mul_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�cs[.��?�Unknown
g�DeviceFill"model_4/lstm/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�,yi��?�Unknown
i�DeviceFill"model_4/lstm_1/zeros(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@Q7u=�܎�>Y�|斤��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_37/ResourceApplyAdam(1R���Q@9R���Q@AR���Q@IR���Q@Q�&�y
@�>Y�o����?�Unknown
W�DeviceMul"Mul(1R���Q@9R���Q@AR���Q@IR���Q@Q�&�y
@�>Y�b���?�Unknown
��DeviceSlice"9gradient_tape/model_4/token_char_hybrid_embedding/Slice_1(1R���Q@9R���Q@AR���Q@IR���Q@Q�&�y
@�>Y�U%T��?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_35/ResourceApplyAdam(1���Mb@9���Mb@A���Mb@I���Mb@Q��,8��>Y.������?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_2(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@Q�:P��S�>Y�ռ����?�Unknown
`�DeviceCast"Adam/Cast_1(11�Zd @91�Zd @A1�Zd @I1�Zd @Q�}9�	��>YAX����?�Unknown
��DeviceAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��w��?�Unknown
��DeviceAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��<��?�Unknown
~�DeviceReadVariableOp"Adam/Adam/update/ReadVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y6qCJd��?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��i����?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�,����?�Unknown
j�DeviceSqrt"Adam/Adam/update/Sqrt(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y+������?�Unknown
j�DeviceAddV2"Adam/Adam/update/add(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y������?�Unknown
j�DeviceMul"Adam/Adam/update/mul_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YyEX)��?�Unknown
j�DeviceMul"Adam/Adam/update/mul_4(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y �)�P��?�Unknown
j�DeviceMul"Adam/Adam/update/mul_5(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y� P*x��?�Unknown
p�DeviceRealDiv"Adam/Adam/update/truediv(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yn^v����?�Unknown
��DeviceResourceApplyAdam"%Adam/Adam/update_31/ResourceApplyAdam(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y������?�Unknown
��DeviceAssignVariableOp"#Adam/Adam/update_9/AssignVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��e���?�Unknown
��DeviceReadVariableOp"!Adam/Adam/update_9/ReadVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Ycw����?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y
�8=��?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�26�d��?�Unknown
��DeviceReadVariableOp"#Adam/Adam/update_9/ReadVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YX�\
���?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��s���?�Unknown
l�DeviceMul"Adam/Adam/update_9/mul_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�K�����?�Unknown
^�DeviceSqrt"	Adam/Sqrt(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YM��E��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y���)��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_10(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�dQ��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_13(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YB�B�x��?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_14(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�i���?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_15(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�}�S���?�Unknown
v�Device	ZerosLike"Adam/gradients/zeros_like_16(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y7۵����?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_4(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�8�%��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_5(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y���=��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_6(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y,�(�d��?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_7(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�QOa���?�Unknown
u�Device	ZerosLike"Adam/gradients/zeros_like_8(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yz�uʳ��?�Unknown
\�DeviceMul"Adam/mul(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y!�3���?�Unknown
^�DeviceSub"
Adam/sub_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�j��?�Unknown
^�DeviceSub"
Adam/sub_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yo��*��?�Unknown
^�DeviceSub"
Adam/sub_3(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y&oQ��?�Unknown
d�DeviceRealDiv"Adam/truediv(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��5�x��?�Unknown
w�DeviceAssignAddVariableOp"AssignAddVariableOp(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yd�[A���?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y?�����?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_3(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y������?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_4(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YY��|��?�Unknown
Y�DeviceCast"Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y X��=��?�Unknown
[�DeviceCast"Cast_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��Oe��?�Unknown
[�DeviceCast"Cast_3(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YNB����?�Unknown
[�DeviceEqual"Equal(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�ph!���?�Unknown
h�Device_Send"IteratorGetNext/_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�Ύ����?�Unknown
h�Device_Recv"IteratorGetNext/_6(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YC,����?�Unknown
h�Device
LogicalAnd"
LogicalAnd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��\*��?�Unknown�
Y�DeviceSum"Sum_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y���Q��?�Unknown
c�DeviceDivNoNan"
div_no_nan(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y8E(/y��?�Unknown
e�DeviceDivNoNan"div_no_nan_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YߢN����?�Unknown
��DeviceSlice"+gradient_tape/model_4/bidirectional_1/Slice(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y� u���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_1/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y-^�j���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_3/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YԻ����?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_4/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y{�<>��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_5/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y"w�e��?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_6/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��4���?�Unknown
�DeviceReluGrad"&gradient_tape/model_4/dense_8/ReluGrad(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yp2[x���?�Unknown
}�DeviceMul")gradient_tape/model_4/dropout/dropout/Mul(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y������?�Unknown
��DeviceSlice";gradient_tape/model_4/token_char_positional_embedding/Slice(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��J��?�Unknown
��DeviceSlice"=gradient_tape/model_4/token_char_positional_embedding/Slice_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YeKγ*��?�Unknown
��DeviceSlice"=gradient_tape/model_4/token_char_positional_embedding/Slice_2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��R��?�Unknown
��DeviceCast"cmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��y��?�Unknown
��DeviceStridedSlice"1model_4/char_vectorizer/StringSplit/strided_slice(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YZdA���?�Unknown
g�DeviceRelu"model_4/dense/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�gX���?�Unknown
o�DeviceBiasAdd"model_4/dense_1/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y������?�Unknown
i�DeviceRelu"model_4/dense_1/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YO}�*��?�Unknown
o�DeviceBiasAdd"model_4/dense_2/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��ړ>��?�Unknown
i�DeviceRelu"model_4/dense_2/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�8�e��?�Unknown
o�DeviceBiasAdd"model_4/dense_3/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>YD�'f���?�Unknown
i�DeviceRelu"model_4/dense_3/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��Mϴ��?�Unknown
o�DeviceBiasAdd"model_4/dense_4/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�Qt8���?�Unknown
i�DeviceRelu"model_4/dense_4/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y9�����?�Unknown
i�DeviceCast"model_4/dense_5/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��
+��?�Unknown
i�DeviceRelu"model_4/dense_5/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�j�sR��?�Unknown
i�DeviceRelu"model_4/dense_6/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y.��y��?�Unknown
i�DeviceRelu"model_4/dense_7/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�%4F���?�Unknown
o�DeviceBiasAdd"model_4/dense_8/BiasAdd(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y|�Z����?�Unknown
i�DeviceRelu"model_4/dense_8/Relu(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y#����?�Unknown
q�DeviceCast"model_4/dropout/dropout/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�>����?�Unknown
��DeviceCast"]model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yq���>��?�Unknown
��DeviceCast"_model_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y��Sf��?�Unknown
��DeviceCast"fmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�W����?�Unknown
��DeviceStridedSlice"4model_4/text_vectorization/StringSplit/strided_slice(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yf�@&���?�Unknown
��DeviceStridedSlice"6model_4/text_vectorization/StringSplit/strided_slice_1(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Yg����?�Unknown
��DeviceSelectV2"3model_4/text_vectorization/string_lookup_1/SelectV2(1����Mb @9����Mb @A����Mb @I����Mb @Qz��.���>Y�p����?�Unknown
^�DevicePow"
Adam/Pow_1(1T㥛�  @9T㥛�  @AT㥛�  @IT㥛�  @Q�T��e�>Y�4�*��?�Unknown
y�DeviceAssignAddVariableOp"AssignAddVariableOp_1(1T㥛�  @9T㥛�  @AT㥛�  @IT㥛�  @Q�T��e�>Y���Q��?�Unknown
��DeviceSlice"-gradient_tape/model_4/bidirectional_1/Slice_1(1X9��v��?9X9��v��?AX9��v��?IX9��v��?Q^]���>Y�#n�w��?�Unknown
��Device_Send"kmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_53(1�������?9�������?A�������?I�������?Qo�*�%��>Y������?�Unknown
��Device_Send"Zmodel_4_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value/_15(1+�����?9+�����?A+�����?I+�����?Q=�a�>Y~�����?�Unknown
��Device_Recv"7categorical_crossentropy/weighted_loss/num_elements/_74(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?Q!R��nf�>Y�E����?�Unknown
��Device_Recv"jmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/_34(1�l�����?9�l�����?A�l�����?I�l�����?Q�$.���>Y�s����?�Unknown
��DeviceAssignVariableOp"%Adam/Adam/update_9/AssignVariableOp_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>Y��so���?�Unknown
\�DeviceSub"Adam/sub(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>Yu�$��?�Unknown
[�DeviceCast"Cast_4(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>YI ����?�Unknown
��DeviceCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>Y/-�+��?�Unknown
��DeviceSlice"7gradient_tape/model_4/token_char_hybrid_embedding/Slice(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>Y�]�A?��?�Unknown
��DeviceCast"\model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?Qz��.���>YŌS�R��?�Unknown
��Device_Send"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_59(1X9��v��?9X9��v��?AX9��v��?IX9��v��?Q^]���>Y""Bf��?�Unknown
^�Device_Recv"Size/_76(1���Q��?9���Q��?A���Q��?I���Q��?QBi��Iy�>Y��x��?�Unknown
��Device_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_19(1���Q��?9���Q��?A���Q��?I���Q��?QBi��Iy�>Y������?�Unknown
��Device_Send"nmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_55(1V-��?9V-��?AV-��?IV-��?Q'�ob���>Y^|{ۜ��?�Unknown
��Device	_HostRecv"gmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast/_38(1�������?9�������?A�������?I�������?Qo�*�%��>Y�x�@���?�Unknown
��Device	_HostRecv"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1/_42(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q7u=�܎�>Y�����?�Unknown
��Device	_HostRecv"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1/_44(1�~j�t��?9�~j�t��?A�~j�t��?I�~j�t��?Q7u=�܎�>Y1?k����?�Unknown
f�Device_Recv"strided_slice/_8(1Zd;�O��?9Zd;�O��?AZd;�O��?IZd;�O��?Q�:P��S�>Y�5����?�Unknown
��Device_Send"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_63(1�I+��?9�I+��?A�I+��?I�I+��?Q� c`J�>Y9Z����?�Unknown
��Device_Send"Tmodel_4_char_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value/_9(1�I+��?9�I+��?A�I+��?I�I+��?Q� c`J�>Y=i���?�Unknown
��Device	_HostRecv"jmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast/_40(1/�$��?9/�$��?A/�$��?I/�$��?Q��u-��>Y�������?�Unknown
^�Device_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_13(Y�������?�Unknown
��Device_Recv"mmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/_36(Y�������?�Unknown
]�Device_Send"4model_4/text_vectorization/string_lookup_1/Equal/_25(Y�������?�Unknown
��HostStringSplitV2"1model_4/char_vectorizer/StringSplit/StringSplitV2(1�K7�A֑@9�K7�A֑@A�K7�A֑@I�K7�A֑@a������?i������?�Unknown
��HostStaticRegexReplace"*model_4/char_vectorizer/StaticRegexReplace(1;�O����@9;�O����@A;�O����@I;�O����@a�05�	��?in����O�?�Unknown
��HostStaticRegexReplace"-model_4/text_vectorization/StaticRegexReplace(1�Q����@9�Q����@A�Q����@I�Q����@a�9ǏG]�?i6�N/��?�Unknown
��HostLookupTableFindV2"Cmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2(1����M"�@9����M"�@A����M"�@I����M"�@arN!�Xͳ?i���^i��?�Unknown
w�HostDataset"!Iterator::Root::Prefetch::BatchV2(1��(\s�@9��(\s�@A7�A`�˃@I7�A`�˃@an���\x�?i."�����?�Unknown
��HostStringSplitV2"4model_4/text_vectorization/StringSplit/StringSplitV2(1�$��B�@9�$��B�@A�$��B�@I�$��B�@acb-����?ic��gx��?�Unknown
}�HostDataset"&Iterator::Root::Prefetch::BatchV2::Zip(�1'1���@9'1��@Ah��|?�~@Ih��|?��?a`�է�b�?iIh��m�?�Unknown
��HostDataset"6Iterator::Root::Prefetch::BatchV2::Zip[0]::TensorSlice(�1���Ʃ}@9���Ʃ�?A���Ʃ}@I���Ʃ�?aBE���,�?i�7Bp@�?�Unknown
��HostLookupTableFindV2"Hmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2(1�Q���w@9�Q���w@A�Q���w@I�Q���w@aVB���3�?i�!�ί��?�Unknown
��HostDataset"6Iterator::Root::Prefetch::BatchV2::Zip[1]::TensorSlice(�1��C�l�u@9��C�l��?A��C�l�u@I��C�l��?a����J�?iZU.Q�?�Unknown
��HostRaggedTensorToTensor";model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor(1� �rh�l@9� �rh�l@A� �rh�l@I� �rh�l@a_�2	4@�?i���R��?�Unknown
�HostEqual"+model_4/char_vectorizer/string_lookup/Equal(1�A`�Жf@9�A`�Жf@A�A`�Жf@I�A`�Жf@a�)u�7�?iZI����?�Unknown
}�HostStringLower"#model_4/char_vectorizer/StringLower(1���Sse@9���Sse@A���Sse@I���Sse@a�����?i���R�D�?�Unknown
��HostBincount"gmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount(1�O��n2e@9�O��n2e@A�O��n2e@I�O��n2e@aC(�=�ؔ?iB�<���?�Unknown
f�Host_Send"IteratorGetNext/_1(1�"��~
_@9�"��~
_@A�"��~
_@I�"��~
_@a�������?i��L�e�?�Unknown
��HostStringLower"&model_4/text_vectorization/StringLower(1�G�z,]@9�G�z,]@A�G�z,]@I�G�z,]@a7y�`��?i�OЀ��?�Unknown
��HostEqual"0model_4/text_vectorization/string_lookup_1/Equal(1^�IV@9^�IV@A^�IV@I^�IV@a/�� ��?i2�R/�?�Unknown
��HostRaggedTensorToTensor">model_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor(1�Zd[U@9�Zd[U@A�Zd[U@I�Zd[U@a[p*F�?i��Tk��?�Unknown
��Host_Send"Gmodel_4/char_vectorizer/string_lookup/None_Lookup/LookupTableFindV2/_23(1V-�S@9V-�S@AV-�S@IV-�S@a���7�?i9��R���?�Unknown
��Host_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_17(1H�z�gN@9H�z�gN@AH�z�gN@IH�z�gN@a��b�}?i����?�Unknown
��Host_Send"8model_4/text_vectorization/StringSplit/StringSplitV2/_19(1��� �rM@9��� �rM@A��� �rM@I��� �rM@a_N��m�|?i�Z��E�?�Unknown
��Host_Send"4model_4/text_vectorization/string_lookup_1/Equal/_25(1H�z�GL@9H�z�GL@AH�z�GL@IH�z�GL@a1[ݍY�{?i9�Y}�?�Unknown
��Host_Send"Lmodel_4/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2/_27(1bX9�K@9bX9�K@AbX9�K@IbX9�K@a8?祡�z?i��P醲�?�Unknown
��Host_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_13(1�����H@9�����H@A�����H@I�����H@a��z>B,x?i���m���?�Unknown
��Host_Send"kmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_53(1�S㥛�G@9�S㥛�G@A�S㥛�G@I�S㥛�G@a��_!w?i ,"�?�Unknown
��Host_Send"Bmodel_4/text_vectorization/RaggedToTensor/RaggedTensorToTensor/_67(1�����=G@9�����=G@A�����=G@I�����=G@a�����v?i��5��>�?�Unknown
��Host_Send"/model_4/char_vectorizer/string_lookup/Equal/_21(1�O��nrF@9�O��nrF@A�O��nrF@I�O��nrF@a�%<��v?iGa&k�?�Unknown
f�Host
LogicalAnd"
LogicalAnd(1=
ףp�E@9=
ףp�E@A=
ףp�E@I=
ףp�E@a�(�-�u?i���w��?�Unknown�
��HostBincount"jmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount(19��v��D@99��v��D@A9��v��D@I9��v��D@aMZ��8t?i5Ҽnu��?�Unknown
��Host_Send"?model_4/char_vectorizer/RaggedToTensor/RaggedTensorToTensor/_65(1X9���C@9X9���C@AX9���C@IX9���C@a֨��Uss?i�}V\��?�Unknown
��Host_Send"5model_4/char_vectorizer/StringSplit/StringSplitV2/_11(1�����LC@9�����LC@A�����LC@I�����LC@ahՍ�\�r?i2���R�?�Unknown
v�HostFlushSummaryWriter"FlushSummaryWriter(1�I+�B@9�I+�B@A�I+�B@I�I+�B@a�?�62hr?i�%8#0�?�Unknown�
��Host_Send"nmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount/_55(1���Mb @@9���Mb @@A���Mb @@I���Mb @@a�a�a��o?i�b��O�?�Unknown
j�HostWriteSummary"WriteSummary(1���S�e7@9���S�e7@A���S�e7@I���S�e7@a�[ok(g?ioB��f�?�Unknown�
��Host_Recv"7model_4/text_vectorization/string_lookup_1/SelectV2/_62(1!�rh�M6@9!�rh�M6@A!�rh�M6@I!�rh�M6@a�w<Vu�e?i�~$��|�?�Unknown
d�HostDataset"Iterator::Root(1X9��vND@9X9��vND@A��|?5�5@I��|?5�5@a9EDh�e?i,�/�o��?�Unknown
��Host_Recv"2model_4/char_vectorizer/string_lookup/SelectV2/_58(1��"���2@9��"���2@A��"���2@I��"���2@aY���ob?i!KM�ߤ�?�Unknown
n�HostDataset"Iterator::Root::Prefetch(1��Q��2@9��Q��2@A��Q��2@I��Q��2@aO=ȯ&Pb?i^��/��?�Unknown
��Host	_HostSend";gradient_tape/model_4/token_embed/embedding_lookup/Size/_71(1��� �r'@9��� �r'@A��� �r'@I��� �r'@a�[v5�W?i�Ηз��?�Unknown
��Host_Recv"jmodel_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum/_48(1���S�!@9���S�!@A���S�!@I���S�!@a�4�ٗQ?i&�����?�Unknown
��Host	_HostSend":gradient_tape/model_4/char_embed/embedding_lookup/Size/_69(1���S�@9���S�@A���S�@I���S�@a�M��(M?i��b����?�Unknown
��Host_Recv"Umodel_4_char_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value/_10(1ˡE���@9ˡE���@AˡE���@IˡE���@a���IևK?i9L����?�Unknown
��Host_Recv"mmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum/_52(1������@9������@A������@I������@a �`Ѹ\J?ic�)G��?�Unknown
f�Host_Send"IteratorGetNext/_3(1u�V@9u�V@Au�V@Iu�V@akT�c�I?ii�",���?�Unknown
m�HostIteratorGetNext"IteratorGetNext(1�C�l��@9�C�l��@A�C�l��@I�C�l��@a�E<���G?izHٔ��?�Unknown
��Host_Recv"^model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_46(1�V�@9�V�@A�V�@I�V�@a�����UD?i��L���?�Unknown
��Host_Recv"Zmodel_4_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value/_16(1V-�@9V-�@AV-�@IV-�@a��h¹k@?iۡe;���?�Unknown
��Host_Recv"cmodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_64(1�G�z@9�G�z@A�G�z@I�G�z@a�]5�6�:?i�H9���?�Unknown
��Host_Recv"`model_4/char_vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/_60(1w��/�@9w��/�@Aw��/�@Iw��/�@ap��v%�4?iD&�&���?�Unknown
��Host_Recv"amodel_4/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast/_50(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?as֨��.?i������?�Unknown
f�Host_Send"IteratorGetNext/_5(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��?��&?i�w�b���?�Unknown
b�HostIdentity"Identity(1��n���?9��n���?A��n���?I��n���?a�K�H�y ?i�������?�Unknown�2Nvidia GPU (Pascal)