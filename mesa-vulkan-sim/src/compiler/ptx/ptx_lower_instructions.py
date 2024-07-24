# Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
# The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution. Neither the name of
# The University of British Columbia nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from enum import unique
from ptx_parser import *
import sys
import os


class Intersection_Table_Type(Enum):
    Baseline = auto()
    FCC = auto()

intersection_table_type = Intersection_Table_Type.Baseline


def vector_suffix_letter(x):
    if x == 0:
        return 'x'
    elif x == 1:
        return 'y'
    elif x == 2:
        return 'z'
    elif x == 3:
        return 'w'

def vector_suffix_number(x):
    if x == 'x':
        return 0
    elif x == 'y':
        return 1
    elif x == 'z':
        return 2
    elif x == 'w':
        return 3



def unwrapp_vector(ptx_shader, vectorVariableName, unwrappedName):
    declaration, _ = ptx_shader.findDeclaration(vectorVariableName)
    assert declaration.isVector()

    newRegNames = [(unwrappedName + '_' + str(i)) for i in range(declaration.vectorSize())]

    newDeclarations = list()
    for i in range(declaration.vectorSize()):
        newDeclaration = PTXDecleration()
        newDeclaration.leadingWhiteSpace = declaration.leadingWhiteSpace
        newDeclaration.buildString(DeclarationType.Register, None, declaration.variableType, newRegNames[i])
        newDeclarations.append(newDeclaration)

    unwrapMovs = list()
    # for i in range(declaration.vectorSize()):
    #     newMov = PTXFunctionalLine()
    #     newMov.leadingWhiteSpace = declaration.leadingWhiteSpace

    #     variableType = declaration.variableType
    #     if variableType == '.b32':
    #         variableType = '.f32'
    #     elif variableType == '.b64':
    #         variableType = '.f64'
    #     zero = '0'
    #     if variableType[1] == 'f':
    #         zero = '0F00000000'
    #     newMov.buildString('add%s' % variableType, (newRegNames[i], vectorVariableName + '.' + vector_suffix_letter(i), zero))
    #     unwrapMovs.append(newMov)
    newMov = PTXFunctionalLine()
    newMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    newMov.buildString('unwrap_32_4', tuple(newRegNames + [vectorVariableName, ]))
    unwrapMovs.append(newMov)
    

    wrapMovs = list()
    # for i in range(declaration.vectorSize()):
    #     wrapMov = PTXFunctionalLine()
    #     wrapMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    #     variableType = declaration.variableType
    #     if variableType == '.b32':
    #         variableType = '.f32'
    #     elif variableType == '.b64':
    #         variableType = '.f64'
    #     zero = '0'
    #     if variableType[1] == 'f':
    #         zero = '0F00000000'
    #     wrapMov.buildString('add%s' % (variableType), (vectorVariableName + '.' + vector_suffix_letter(i), newRegNames[i], zero))
    #     wrapMovs.append(wrapMov)
    wrapMov = PTXFunctionalLine()
    wrapMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    wrapMov.buildString('wrap_32_4', tuple([vectorVariableName, ] + newRegNames))
    wrapMovs.append(wrapMov)

    return newRegNames, newDeclarations, unwrapMovs, wrapMovs


def translate_vector_operands(ptx_shader, unique_ID):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        if 'undefined' in line.comment and \
            ('vec2' in line.comment or \
            'vec3' in line.comment or \
            'vec4' in line.comment):
            newLines = list()

            newLine = PTXFunctionalLine()
            if line.fullFunction == FunctionalType.load_const:
                line.fullFunction = 'mov.f32'
            newLine.leadingWhiteSpace = line.leadingWhiteSpace
            argIndex = 0
            arg = line.args[argIndex]
            vectorRegName = line.args[0]
            newRegName = vectorRegName + '_' + str(0)
            args = line.args
            args = list(args)
            args[argIndex] = newRegName
            newLine.buildString(line.fullFunction, args)
            newLines.append(newLine)

            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = line.leadingWhiteSpace
            newRegName = vectorRegName + '_' + str(1)
            args[argIndex] = newRegName
            newLine.buildString(line.fullFunction, args)
            newLines.append(newLine)

            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = line.leadingWhiteSpace
            newRegName = vectorRegName + '_' + str(2)
            args[argIndex] = newRegName
            newLine.buildString(line.fullFunction, args)
            newLines.append(newLine)
            
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = line.leadingWhiteSpace
            newRegName = vectorRegName + '_' + str(3)
            args[argIndex] = newRegName
            newLine.buildString(line.fullFunction, args)
            newLines.append(newLine)

            ptx_shader.lines.remove(line)
            ptx_shader.lines[index:index] = newLines

        elif line.instructionClass == InstructionClass.Functional:
            # print("#######################")
            print(line.fullLine)

            for argIndex in range(len(line.args)):
                arg = line.args[argIndex]
                if '.' not in arg:
                    continue
                # if '.field0' in arg:
                #     continue
                assert arg[-2] == '.'
                vectorRegName = arg[:-2]
                newRegName = vectorRegName + '_' + str(vector_suffix_number(arg[-1]))

                args = line.args
                args[argIndex] = newRegName
                line.buildString(line.fullFunction, args)
        
        elif line.instructionClass == InstructionClass.VariableDeclaration:
            if line.declarationType != DeclarationType.Register:
                continue
            if not line.isVector():
                continue

            # print("#######################")
            # print(line.fullLine)

            newLines = list()
            for i in range(line.vectorSize()):
                newLine = PTXDecleration()
                newLine.leadingWhiteSpace = line.leadingWhiteSpace
                newLine.buildString(line.declarationType, None, line.variableType, line.variableName + '_' + str(i))
                newLines.append(newLine)
            
            ptx_shader.lines.remove(line)
            ptx_shader.lines[index:index] = newLines



def translate_descriptor_set_instructions(ptx_shader):
    for line in ptx_shader.lines:
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.vulkan_resource_index:
            dst, whatIsThis, descSet, binding, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString(FunctionalType.load_vulkan_descriptor, (dst, descSet, binding, type))
        
        elif line.functionalType == FunctionalType.load_vulkan_descriptor:
            dst, src, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString('mov.b64', (dst, src))

def translate_deref_instructions(ptx_shader):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.deref_cast:
            dst, baseType, src, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            line.buildString('mov.b64', (dst, src))
        
        elif line.functionalType == FunctionalType.deref_struct:
            dst, need_deref, src, ptr, fieldName, offset, baseType, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            line.buildString('add.u64', (dst, src, offset))
        
        elif line.functionalType == FunctionalType.deref_array:
            dst, need_deref, src, arrayIndex, arrayStride, baseType, type = line.args

            if baseType == 'descriptor':
                arrayStride = str(32)

            assert int(arrayStride) != 0

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            if arrayIndex[0] != '%': # const array index
                line.buildString('add.u64', (dst, src, str(int(arrayIndex) * int(arrayStride))))
            else: # reg array index

                # exit(-1)
                newLines = list()

                indexRegName_32 = dst + '_array_index_32'
                indexRegName_64 = dst + '_array_index_64'

                newDeclaration_32 = PTXDecleration()
                newDeclaration_32.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDeclaration_32.buildString(DeclarationType.Register, None, '.u32', indexRegName_32)

                newDeclaration_64 = PTXDecleration()
                newDeclaration_64.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDeclaration_64.buildString(DeclarationType.Register, None, '.u64', indexRegName_64)

                indexDeclaration, _ = ptx_shader.findDeclaration(arrayIndex)

                newSet = PTXFunctionalLine()
                newSet.leadingWhiteSpace = declaration.leadingWhiteSpace
                if indexDeclaration.variableType == '.u32':
                    newSet.buildString('mov.u32', (indexRegName_32, arrayIndex))
                else:
                    variableType = indexDeclaration.variableType
                    if variableType[1] == 'b':
                        variableType = '.u' + variableType[2:]
                    if variableType[1] == 'f':
                        # newSet.buildString('cvt.rni.u32%s' % variableType, (indexRegName_32, arrayIndex))
                        newSet.buildString('mov.b32', (indexRegName_32, arrayIndex))
                    else:
                        newSet.buildString('cvt.u32%s' % variableType, (indexRegName_32, arrayIndex))


                newMult = PTXFunctionalLine()
                newMult.leadingWhiteSpace = declaration.leadingWhiteSpace
                newMult.buildString('mul.wide.u32', (indexRegName_64, indexRegName_32, arrayStride))

                newAdd = PTXFunctionalLine()
                newAdd.leadingWhiteSpace = declaration.leadingWhiteSpace
                newAdd.comment = line.comment
                newAdd.buildString('add.u64', (dst, src, indexRegName_64))

                ptx_shader.lines.remove(line)
                ptx_shader.lines[index:index] = (newDeclaration_32, newDeclaration_64, newSet, newMult, newAdd)

        
        elif line.functionalType == FunctionalType.load_deref:
            vectorCount, dst, ptr, access = line.args

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            srcDeclaration, _ = ptx_shader.findDeclaration(ptr)
            
            # assert srcDeclaration.pointerVariableType == declaration.variableType

            # print(line.fullLine)

            # assert srcDeclaration.pointerVariableType[:2] == '.b' or srcDeclaration.pointerVariableType[:2] == '.f'

            assert srcDeclaration.pointerVariableType[2:] == declaration.variableType[2:]
            declaration.buildString(declaration.declarationType, declaration.vector, srcDeclaration.pointerVariableType, declaration.variableName)

            # if ptr == '%ssa_13':
            #     print(declaration.fullLine)
            #     exit(-1)

            
            # if srcDeclaration.pointerVariableType == None:
            #     exit(-1)

            if not declaration.isVector():
                line.buildString('ld.global%s' % (declaration.variableType), (dst, '[%s]' % ptr))
            else:
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    if int(vectorCount) > 0 and i >= int(vectorCount):
                        break
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    # print('#432 ' + declaration.fullLine)
                    newFunctional.buildString('ld.global%s' % declaration.variableType, (declaration.variableName + '_' + str(i), '[' + ptr + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']'))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines.remove(line)
                ptx_shader.lines[index: index] = newLines
        
        
        elif line.functionalType == FunctionalType.store_deref:
            # print("################")
            # print(line.fullLine)
            ptr, dst, wrmask, access = line.args
            # dst = line.args[1]
            # ptr = line.args[0]

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            if not declaration.isVector():
                line.buildString('st.global%s' % (declaration.variableType), ('[%s]' % ptr, dst))
            else:
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    if int(wrmask) & (1 << i) == 0:
                        continue
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('st.global%s' % declaration.variableType, ('[' + ptr  + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']', declaration.variableName + '_' + str(i)))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                ptx_shader.lines.remove(line)
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines[index: index] = newLines
        

        elif line.functionalType == FunctionalType.deref_var:
            dst, src, baseType, type = line.args
            
            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            assert not declaration.isVector()
            assert declaration.declarationType == DeclarationType.Register
            declaration.buildString(DeclarationType.Register, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            line.buildString('mov.b64', (dst, '%' + src))


        # elif line.functionalType == FunctionalType.mov:
        #     print(line.fullLine)
        #     print(line.args)
        #     #assert len(line.args) == 2
        #     if '.' in line.args[0]: #TODO: args with brackets are parsed incorrectly
        #         if line.vector == None:
        #             variableType = line.variableType
        #             print(line.fullLine)
        #             print(variableType)
        #             #exit(-1)
        #             if variableType == '.b32':
        #                 variableType = '.f32'
        #             elif variableType == '.b64':
        #                 variableType = '.f64'
        #             zero = '0'
        #             if variableType[1] == 'f':
        #                 zero = '0F00000000'
        #             line.buildString('add%s' % variableType, (line.args[0], line.args[1], zero))

def translate_trace_ray(ptx_shader, shaderIDs):
    trace_ray_ID = 0
    skip_lines = -1
    for index in range(len(ptx_shader.lines)):
        if index <= skip_lines:
            continue
        line = ptx_shader.lines[index]
        print(line)
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.trace_ray:
            continue

        assert len(line.args) == 11


        traversal_finished_reg = '%traversal_finished_' + str(trace_ray_ID)
        traversal_finished_declaration = PTXDecleration()
        traversal_finished_declaration.leadingWhiteSpace = line.leadingWhiteSpace
        traversal_finished_declaration.buildString(DeclarationType.Register, None, '.u32', traversal_finished_reg)


        topLevelAS, rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, origin, Tmin, direction, Tmax, payload = line.args
        line.args = line.args[:-1] # MRS_TODO: why there is a payload (in glsl code it is NULL but translated to ssa_88)? and why it gets an error to run?
        args = line.args

        # originRegNames, originDeclarations, originMovs, _ = unwrapp_vector(ptx_shader, origin, "trace_ray_" + str(index) + "_origin")
        # directionRegNames, directionDeclarations, directionMovs, _ = unwrapp_vector(ptx_shader, direction, "trace_ray_" + str(index) + "_direction")

        originRegNames = [origin + '_' + str(i) for i in range(3)]
        directionRegNames = [direction + '_' + str(i) for i in range(3)]

        args[8:9] = directionRegNames[:3]
        args[6:7] = originRegNames[:3]
        args.append(traversal_finished_reg)
        line.buildString(line.functionalType, args)

        
        #intersection shaders
        intersection_lines = []
        if ShaderType.Intersection in shaderIDs:

            intersection_counter_reg = '%intersection_counter_' + str(trace_ray_ID)
            intersection_counter_declaration = PTXDecleration()
            intersection_counter_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_counter_declaration.buildString(DeclarationType.Register, None, '.u32', intersection_counter_reg)
            intersection_lines.append(intersection_counter_declaration)

            intersection_counter_mov = PTXFunctionalLine()
            intersection_counter_mov.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_counter_mov.buildString('mov.u32', (intersection_counter_reg, '0'))
            intersection_lines.append(intersection_counter_mov)

            intersection_loop_label_str = 'intersection_loop_' + str(trace_ray_ID)
            intersection_loop_label = PTXLine('')
            intersection_loop_label.fullLine = line.leadingWhiteSpace + intersection_loop_label_str + ':\n'
            intersection_lines.append(intersection_loop_label)

            intersection_exit_reg = '%intersections_exit_' + str(trace_ray_ID)
            intersection_exit_declaration = PTXDecleration()
            intersection_exit_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_exit_declaration.buildString(DeclarationType.Register, None, '.pred', intersection_exit_reg)
            intersection_lines.append(intersection_exit_declaration)

            intersection_exit = PTXFunctionalLine()
            intersection_exit.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_exit.buildString('intersection_exit.pred', (intersection_exit_reg, intersection_counter_reg, traversal_finished_reg))
            intersection_lines.append(intersection_exit)

            exit_intersection_label_str = 'exit_intersection_label_' + str(trace_ray_ID)
            exit_intersection_bra = PTXFunctionalLine()
            exit_intersection_bra.leadingWhiteSpace = line.leadingWhiteSpace
            exit_intersection_bra.condition = '@' + intersection_exit_reg
            exit_intersection_bra.buildString(FunctionalType.bra, (exit_intersection_label_str, ))
            intersection_lines.append(exit_intersection_bra)

            shader_data_address_reg = '%shader_data_address_' + str(trace_ray_ID)
            shader_data_address_declaration = PTXDecleration()
            shader_data_address_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            shader_data_address_declaration.buildString(DeclarationType.Register, None, '.b64', shader_data_address_reg)
            intersection_lines.append(shader_data_address_declaration)

            get_shader_data_address = PTXFunctionalLine()
            get_shader_data_address.leadingWhiteSpace = line.leadingWhiteSpace
            get_shader_data_address.buildString('get_intersection_shader_data_address', (shader_data_address_reg, intersection_counter_reg))
            intersection_lines.append(get_shader_data_address)


            if intersection_table_type == Intersection_Table_Type.FCC:
                run_intersection_reg = '%run_intersection_' + str(trace_ray_ID)
                run_intersection_declaration = PTXDecleration()
                run_intersection_declaration.leadingWhiteSpace = line.leadingWhiteSpace
                run_intersection_declaration.buildString(DeclarationType.Register, None, '.pred', run_intersection_reg)
                intersection_lines.append(run_intersection_declaration)

                run_intersection = PTXFunctionalLine()
                run_intersection.leadingWhiteSpace = line.leadingWhiteSpace
                run_intersection.buildString('run_intersection.pred', (run_intersection_reg, intersection_counter_reg, traversal_finished_reg))
                intersection_lines.append(run_intersection)

                skip_intersection_label_str = 'skip_intersection_label_' + str(trace_ray_ID)
                skip_intersection_bra = PTXFunctionalLine()
                skip_intersection_bra.leadingWhiteSpace = line.leadingWhiteSpace
                skip_intersection_bra.condition = '@!' + run_intersection_reg
                skip_intersection_bra.buildString(FunctionalType.bra, (skip_intersection_label_str, ))
                intersection_lines.append(skip_intersection_bra)

                primitiveID_reg = '%primitiveID_' + str(trace_ray_ID)
                primitiveID_declaration = PTXDecleration()
                primitiveID_declaration.leadingWhiteSpace = line.leadingWhiteSpace
                primitiveID_declaration.buildString(DeclarationType.Register, None, '.u32', primitiveID_reg)
                intersection_lines.append(primitiveID_declaration)

                primitiveID_load = PTXFunctionalLine()
                primitiveID_load.leadingWhiteSpace = line.leadingWhiteSpace
                primitiveID_load.buildString('ld.global.u32', (primitiveID_reg, '[' + shader_data_address_reg + ']'))
                intersection_lines.append(primitiveID_load)

                instanceID_reg = '%instanceID_' + str(trace_ray_ID)
                instanceID_declaration = PTXDecleration()
                instanceID_declaration.leadingWhiteSpace = line.leadingWhiteSpace
                instanceID_declaration.buildString(DeclarationType.Register, None, '.u32', instanceID_reg)
                intersection_lines.append(instanceID_declaration)

                instanceID_load = PTXFunctionalLine()
                instanceID_load.leadingWhiteSpace = line.leadingWhiteSpace
                instanceID_load.buildString('ld.global.u32', (instanceID_reg, '[' + shader_data_address_reg + ' + 4]'))
                intersection_lines.append(instanceID_load)

                call_intersection = PTXFunctionalLine()
                call_intersection.leadingWhiteSpace = line.leadingWhiteSpace
                call_intersection.buildString(FunctionalType.call_intersection_shader, (intersection_counter_reg, ))
                intersection_lines.append(call_intersection)

                skip_intersection_label = PTXLine('')
                skip_intersection_label.fullLine = line.leadingWhiteSpace + skip_intersection_label_str + ':\n'
                intersection_lines.append(skip_intersection_label)
            
            else: # baseline
                intersection_shaderID_reg = '%intersection_shaderID_' + str(trace_ray_ID)
                intersection_shaderID_declaration = PTXDecleration()
                intersection_shaderID_declaration.leadingWhiteSpace = line.leadingWhiteSpace
                intersection_shaderID_declaration.buildString(DeclarationType.Register, None, '.u32', intersection_shaderID_reg)
                intersection_lines.append(intersection_shaderID_declaration)

                get_intersection_shaderID = PTXFunctionalLine()
                get_intersection_shaderID.leadingWhiteSpace = line.leadingWhiteSpace
                get_intersection_shaderID.buildString(FunctionalType.get_intersection_shaderID, (intersection_shaderID_reg, intersection_counter_reg))
                intersection_lines.append(get_intersection_shaderID)

                for shaderID in shaderIDs[ShaderType.Intersection]:
                    skip_intersection_reg = '%skip_intersection_' + str(shaderID) + '_' + str(trace_ray_ID)
                    skip_intersection_declaration = PTXDecleration()
                    skip_intersection_declaration.leadingWhiteSpace = line.leadingWhiteSpace
                    skip_intersection_declaration.buildString(DeclarationType.Register, None, '.pred', skip_intersection_reg)
                    intersection_lines.append(skip_intersection_declaration)

                    skip_intersection_pred = PTXFunctionalLine()
                    skip_intersection_pred.leadingWhiteSpace = line.leadingWhiteSpace
                    skip_intersection_pred.buildString('setp.ne.u32', (skip_intersection_reg, intersection_shaderID_reg, str(shaderID)))
                    intersection_lines.append(skip_intersection_pred)

                    skip_intersection_label_str = 'skip_intersection_label_' + str(shaderID) + '_' + str(trace_ray_ID)
                    skip_intersection_bra = PTXFunctionalLine()
                    skip_intersection_bra.leadingWhiteSpace = line.leadingWhiteSpace
                    skip_intersection_bra.condition = '@' + skip_intersection_reg
                    skip_intersection_bra.buildString(FunctionalType.bra, (skip_intersection_label_str, ))
                    intersection_lines.append(skip_intersection_bra)

                    call_intersection = PTXFunctionalLine()
                    call_intersection.leadingWhiteSpace = line.leadingWhiteSpace
                    call_intersection.buildString(FunctionalType.call_intersection_shader, (intersection_counter_reg, ))
                    intersection_lines.append(call_intersection)

                    skip_intersection_label = PTXLine('')
                    skip_intersection_label.fullLine = line.leadingWhiteSpace + skip_intersection_label_str + ':\n'
                    intersection_lines.append(skip_intersection_label)

            intersection_counter_add = PTXFunctionalLine()
            intersection_counter_add.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_counter_add.buildString('add.u32', (intersection_counter_reg, intersection_counter_reg, '1'))
            intersection_lines.append(intersection_counter_add)

            intersection_loop_bra = PTXFunctionalLine()
            intersection_loop_bra.leadingWhiteSpace = line.leadingWhiteSpace
            intersection_loop_bra.buildString(FunctionalType.bra, (intersection_loop_label_str, ))
            intersection_lines.append(intersection_loop_bra)

            exit_intersection_label = PTXLine('')
            exit_intersection_label.fullLine = line.leadingWhiteSpace + exit_intersection_label_str + ':\n'
            intersection_lines.append(exit_intersection_label)

        # get hit_geometry
        hit_geometry_reg = '%hit_geometry_' + str(trace_ray_ID)
        hit_geometry_declaration = PTXDecleration()
        hit_geometry_declaration.leadingWhiteSpace = line.leadingWhiteSpace
        hit_geometry_declaration.buildString(DeclarationType.Register, None, '.pred', hit_geometry_reg)

        hit_geometry = PTXFunctionalLine()
        hit_geometry.leadingWhiteSpace = line.leadingWhiteSpace
        hit_geometry.buildString('hit_geometry.pred', (hit_geometry_reg, traversal_finished_reg))

        # closest hit shader
        closest_hit_lines = []

        exit_closest_hit_label_str = 'exit_closest_hit_label_' + str(trace_ray_ID)
        call_closest_hit_bra = PTXFunctionalLine()
        call_closest_hit_bra.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit_bra.condition = '@!' + hit_geometry_reg
        call_closest_hit_bra.buildString(FunctionalType.bra, (exit_closest_hit_label_str, ))
        closest_hit_lines.append(call_closest_hit_bra)

        closest_hit_shaderID_reg = '%closest_hit_shaderID_' + str(trace_ray_ID)
        closest_hit_shaderID_declaration = PTXDecleration()
        closest_hit_shaderID_declaration.leadingWhiteSpace = line.leadingWhiteSpace
        closest_hit_shaderID_declaration.buildString(DeclarationType.Register, None, '.u32', closest_hit_shaderID_reg)
        closest_hit_lines.append(closest_hit_shaderID_declaration)

        get_closest_hit_shaderID = PTXFunctionalLine()
        get_closest_hit_shaderID.leadingWhiteSpace = line.leadingWhiteSpace
        get_closest_hit_shaderID.buildString(FunctionalType.get_closest_hit_shaderID, (closest_hit_shaderID_reg, ))
        closest_hit_lines.append(get_closest_hit_shaderID)

        for shaderID in shaderIDs[ShaderType.Closest_hit]:
            skip_closest_hit_reg = '%skip_closest_hit_' + str(shaderID) + '_' + str(trace_ray_ID)
            skip_closest_hit_declaration = PTXDecleration()
            skip_closest_hit_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            skip_closest_hit_declaration.buildString(DeclarationType.Register, None, '.pred', skip_closest_hit_reg)
            closest_hit_lines.append(skip_closest_hit_declaration)

            skip_closest_hit_pred = PTXFunctionalLine()
            skip_closest_hit_pred.leadingWhiteSpace = line.leadingWhiteSpace
            skip_closest_hit_pred.buildString('setp.ne.u32', (skip_closest_hit_reg, closest_hit_shaderID_reg, str(shaderID)))
            closest_hit_lines.append(skip_closest_hit_pred)

            skip_closest_hit_label_str = 'skip_closest_hit_label_' + str(shaderID) + '_' + str(trace_ray_ID)
            skip_closest_hit_bra = PTXFunctionalLine()
            skip_closest_hit_bra.leadingWhiteSpace = line.leadingWhiteSpace
            skip_closest_hit_bra.condition = '@' + skip_closest_hit_reg
            skip_closest_hit_bra.buildString(FunctionalType.bra, (skip_closest_hit_label_str, ))
            closest_hit_lines.append(skip_closest_hit_bra)

            call_closest_hit = PTXFunctionalLine()
            call_closest_hit.leadingWhiteSpace = line.leadingWhiteSpace
            call_closest_hit.buildString(FunctionalType.call_closest_hit_shader, (str(shaderID), ))
            closest_hit_lines.append(call_closest_hit)

            skip_closest_hit_label = PTXLine('')
            skip_closest_hit_label.fullLine = line.leadingWhiteSpace + skip_closest_hit_label_str + ':\n'
            closest_hit_lines.append(skip_closest_hit_label)
        
        exit_closest_hit_label = PTXLine('')
        exit_closest_hit_label.fullLine = line.leadingWhiteSpace + exit_closest_hit_label_str + ':\n'
        closest_hit_lines.append(exit_closest_hit_label)

        # miss shader
        skip_miss_label_str = 'skip_miss_label_' + str(trace_ray_ID)
        call_miss_bra = PTXFunctionalLine()
        call_miss_bra.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss_bra.condition = '@' + hit_geometry_reg
        call_miss_bra.buildString(FunctionalType.bra, (skip_miss_label_str, ))

        call_miss = PTXFunctionalLine()
        call_miss.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss.buildString(FunctionalType.call_miss_shader, ())

        skip_miss_label = PTXLine('')
        skip_miss_label.fullLine = line.leadingWhiteSpace + skip_miss_label_str + ':\n'

        # finish trace ray
        end_trace_ray = PTXFunctionalLine()
        end_trace_ray.leadingWhiteSpace = line.leadingWhiteSpace
        end_trace_ray.buildString(FunctionalType.end_trace_ray, ())

        newLines = [traversal_finished_declaration, line, PTXLine('\n')]
        newLines.extend(intersection_lines)
        newLines.append(PTXLine('\n'))
        newLines.extend([hit_geometry_declaration, hit_geometry, PTXLine('\n')])
        newLines.extend(closest_hit_lines)
        newLines.append(PTXLine('\n'))
        newLines.extend([call_miss_bra, call_miss, skip_miss_label, PTXLine('\n'), end_trace_ray])


        ptx_shader.lines[index:index + 1] = newLines
        
        skip_lines = index + len(newLines) - 1

        trace_ray_ID += 1



def translate_decl_var(ptx_shader):
    new_declerations = []
    old_declerations = []

    # newReg = PTXDecleration()
    # newReg.leadingWhiteSpace = '\t'
    # newReg.buildString(DeclarationType.Register, None, '.u32', '%allocasize')
    # new_declerations.append(newReg)


    for line in ptx_shader.lines:
        if line.instructionClass != InstructionClass.Functional:
            continue
        if line.functionalType != FunctionalType.decl_var:
            continue

        # print(line.fullLine)
        # print(line.args)
        # exit(-1)

        name, size, vector_number, variable_type, storage_qualifier_type, driver_location, binding, drivername = line.args
        name = '%' + name
        # if int(vector_number) > 1:
        #     continue

        newReg = PTXDecleration()
        newReg.leadingWhiteSpace = '\t'
        newReg.buildString(DeclarationType.Register, None, '.b64', name)

        # if int(vector_number) != 0:
        #     allocation_size = int(size) * int(vector_number)
        # else:
        allocation_size = int(size)


        if int(storage_qualifier_type) == 16: ## uniform type
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('load_vulkan_descriptor', (name, driver_location, binding))
        else:
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('rt_alloc_mem', (name, str(allocation_size), str(storage_qualifier_type), drivername))

        new_declerations.append(newReg)
        # new_declerations.append(newSizeSet)
        new_declerations.append(newLine)
        old_declerations.append(line)
    
    for decleration in old_declerations:
        ptx_shader.lines.remove(decleration)
    
    new_declerations.append(PTXLine('\n'))
    ptx_shader.addToStart(new_declerations)
    
    # for index in range(len(ptx_shader.lines)):
    #     line = ptx_shader.lines[index]
    #     if line.instructionClass != InstructionClass.EntryPoint:
    #         continue
    #     if 'main' not in line.fullLine:
    #         continue
        
    #     index += 1
    #     for decleration in new_declerations:
    #         ptx_shader.lines.insert(index + 1, decleration)
    #         index += 1
    #     break


def translate_load_GL_instructions(ptx_shader):
    skip_lines = -1
    for index in range(len(ptx_shader.lines)):
        if index <= skip_lines:
            continue
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.load_ray_launch_id or line.functionalType == FunctionalType.load_ray_launch_size:
            dst = line.args[0]

            declaration, _ = ptx_shader.findDeclaration(dst)
            assert declaration.isVector()

            newRegNames = [(dst + '_' + str(i)) for i in range(4)]

            # newDeclarations = list()
            # for i in range(4):
            #     newDeclaration = PTXDecleration()
            #     newDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
            #     newDeclaration.buildString(DeclarationType.Register, None, declaration.variableType, newRegNames[i])
            #     newDeclarations.append(newDeclaration)

            # comment = line.comment
            # line.comment = ""
            line.buildString(line.functionalType, (newRegNames[:3]))

            # loadZero = PTXFunctionalLine()
            # loadZero.leadingWhiteSpace = line.leadingWhiteSpace
            # loadZero.buildString('mov%s' % (declaration.variableType), (newRegNames[3], "0"))
            
            # _, _, _, wrapMovs = unwrapp_vector(ptx_shader, declaration.variableName, declaration.variableName)
            # movLine = PTXFunctionalLine()
            # movLine.leadingWhiteSpace = line.leadingWhiteSpace
            # movLine.comment = comment
            # movLine.buildString('mov%s%s' % (declaration.vector, declaration.variableType), (declaration.variableName, '{' + ", ".join(newRegNames) + '}'))

            # ptx_shader.lines[index:index] =  newDeclarations
            # ptx_shader.lines.insert(index + 5, loadZero)
            # ptx_shader.lines[index + 6: index + 6] = wrapMovs[:3]
            # skip_lines = index + 7
        
        
        elif line.functionalType == FunctionalType.load_ray_world_to_object or line.functionalType == FunctionalType.load_ray_object_to_world:
            dst, loadIndex = line.args

            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)


            address_reg = str(dst) + '_address'
            address_declaration = PTXDecleration()
            address_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            address_declaration.buildString(DeclarationType.Register, None, '.b64', address_reg)

            offset = 0
            loads = []
            for regNames in newRegNames:
                newLoad = PTXFunctionalLine()
                newLoad.leadingWhiteSpace = line.leadingWhiteSpace
                newLoad.buildString('ld.global.f32', (regNames, '[' + address_reg + ' + ' + str(offset) + ']'))
                loads.append(newLoad)
                offset += 4

            line.buildString(line.functionalType, [address_reg, loadIndex, ])

            ptx_shader.lines[index:index + 1] = [address_declaration, line] + loads

            skip_lines = index + 2
        

        elif line.functionalType == FunctionalType.load_ray_world_direction:
            dst = line.args[0]

            address_reg = str(dst) + '_address'
            address_declaration = PTXDecleration()
            address_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            address_declaration.buildString(DeclarationType.Register, None, '.b64', address_reg)

            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)

            offset = 0
            loads = []
            for regNames in newRegNames:
                newLoad = PTXFunctionalLine()
                newLoad.leadingWhiteSpace = line.leadingWhiteSpace
                newLoad.buildString('ld.global.f32', (regNames, '[' + address_reg + ' + ' + str(offset) + ']'))
                loads.append(newLoad)
                offset += 4

            line.buildString(line.functionalType, (address_reg, ))

            ptx_shader.lines[index:index + 1] = [address_declaration, line] + loads

            skip_lines = index + 2
        

        elif line.functionalType == FunctionalType.load_ray_world_origin:
            dst = line.args[0]

            address_reg = str(dst) + '_address'
            address_declaration = PTXDecleration()
            address_declaration.leadingWhiteSpace = line.leadingWhiteSpace
            address_declaration.buildString(DeclarationType.Register, None, '.b64', address_reg)

            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)

            offset = 0
            loads = []
            for regNames in newRegNames:
                newLoad = PTXFunctionalLine()
                newLoad.leadingWhiteSpace = line.leadingWhiteSpace
                newLoad.buildString('ld.global.f32', (regNames, '[' + address_reg + ' + ' + str(offset) + ']'))
                loads.append(newLoad)
                offset += 4
            
            line.buildString(line.functionalType, (address_reg, ))

            ptx_shader.lines[index:index + 1] = [address_declaration, line] + loads

            skip_lines = index + 2







def translate_image_deref(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]

        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.image_deref_store:
            image, arg2, arg3, hitValue, arg5, arg6, arg7 = line.args
            args = line.args
            args[3:4] = [(hitValue + '_' + str(i)) for i in range(4)]
            args[1:2] = [(arg2 + '_' + str(i)) for i in range(4)]
            line.buildString(line.functionalType, args)
        
        elif line.functionalType == FunctionalType.image_deref_load:
            dst, image, location, arg3, arg4, arg5, arg6 = line.args
            args = [image, dst, location, arg3, arg4, arg5, arg6]
            dstRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            locationRegNames, _, _, _ = unwrapp_vector(ptx_shader, location, location)
            args[2:3] = locationRegNames
            args[1:2] = dstRegNames
            line.buildString(line.functionalType, args)



def translate_exit(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]

        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.exit:
            continue

        line.buildString(FunctionalType.ret, ())


def translate_phi(ptx_shader):
    nextIndex = 0
    while nextIndex < len(ptx_shader.lines):
        index = nextIndex
        nextIndex += 1
    # for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.phi:
            continue

        print(line.fullLine)

        if len(line.args) == 5:
            dst, blockName0, src0, blockName1, src1 = line.args
        elif len(line.args) == 7:
            dst, blockName0, src0, blockName1, src1, blockName2, src2 = line.args
        
        dstDecleration, dstIndex = ptx_shader.findDeclaration(dst)
        print(src0)
        print(src1)
        src0Decleration, _ = ptx_shader.findDeclaration(src0)
        src1Decleration, _ = ptx_shader.findDeclaration(src1)

        if len(line.args) == 7:
            src2Decleration, _ = ptx_shader.findDeclaration(src2)

        if src0Decleration.variableType == src1Decleration.variableType:
            variableType = src0Decleration.variableType
        else: # this happens because of load_const types are unknown
            if src0Decleration.variableType[0:2] == '.f':
                variableType = src1Decleration.variableType
            elif src1Decleration.variableType[0:2] == '.f':
                variableType = src0Decleration.variableType
            elif src0Decleration.variableType[0:2] == '.u' and src1Decleration.variableType[0:2] == '.s':
                variableType = src1Decleration.variableType #lets go with .s for now
            elif src0Decleration.variableType[0:2] == '.b' and src1Decleration.variableType[0:2] == '.s':
                variableType = src1Decleration.variableType
            elif src0Decleration.variableType[0:2] == '.b' and src1Decleration.variableType[0:2] == '.u':
                variableType = src1Decleration.variableType
            elif src0Decleration.variableType[0:2] == '.u' and src1Decleration.variableType[0:2] == '.b':
                variableType = src0Decleration.variableType
            elif src0Decleration.variableType[0:2] == '.p' or src1Decleration.variableType[0:2] == '.p':
                variableType = '.pred'
            else:
                assert 0

        dstDecleration.buildString(dstDecleration.declarationType, dstDecleration.vector, variableType, dstDecleration.variableName)

        src0Mov = PTXFunctionalLine()
        src0Mov.leadingWhiteSpace = src0Decleration.leadingWhiteSpace
        src0Mov.comment = line.comment
        src0Mov.buildString('mov%s' % variableType, (dst, src0))

        src1Mov = PTXFunctionalLine()
        src1Mov.leadingWhiteSpace = src1Decleration.leadingWhiteSpace
        src1Mov.comment = line.comment
        src1Mov.buildString('mov%s' % variableType, (dst, src1))

        if len(line.args) == 7:
            src2Mov = PTXFunctionalLine()
            src2Mov.leadingWhiteSpace = src2Decleration.leadingWhiteSpace
            src2Mov.comment = line.comment
            src2Mov.buildString('mov%s' % variableType, (dst, src2))


        ptx_shader.lines.remove(dstDecleration)
        ptx_shader.lines.remove(line)

        ptx_shader.addToStart((dstDecleration, PTXLine('\n')))
        ptx_shader.addToEndOfBlock((src0Mov, ), blockName0)
        ptx_shader.addToEndOfBlock((src1Mov, ), blockName1)

        if len(line.args) == 7:
            ptx_shader.addToEndOfBlock((src2Mov, ), blockName2)


def translate_load_const(ptx_shader):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.load_const:
            continue

        dst, const = line.args
        declaration, _ = ptx_shader.findDeclaration(dst)
        declaration.isLoadConst = True

        line.buildString("mov%s" % (declaration.variableType), (dst, const))


        newVariableType = '.b' + declaration.variableType[2:]

        newDeclaration = PTXDecleration()
        newDeclaration.leadingWhiteSpace = declaration.leadingWhiteSpace
        newDeclaration.buildString(DeclarationType.Register, None, newVariableType, dst + '_bits')

        newMov = PTXFunctionalLine()
        newMov.leadingWhiteSpace = line.leadingWhiteSpace
        newMov.buildString("mov%s" % (declaration.variableType), (dst + '_bits', const))

        ptx_shader.lines[index + 1:index + 1] = (newDeclaration, newMov)




def translate_const_operands(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.command[:3] == 'mov':

            movType = line.command[3:]

            dst, src = line.args

            print(line.fullLine)
            
            if src[0] == '%':
                declaration, _ = ptx_shader.findDeclaration(src)                

                if movType[:2] != '.f' and declaration.isLoadConst:
                    line.buildString(line.command.split()[0], (dst, src + '_bits'))
        
        elif line.command[:4] == 'setp':
            dst, src1, src2 = line.args

            type = '.' + line.command.split()[0].split('.')[2]

            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))
        
        elif line.command[:3] == 'add':
            dst, src1, src2 = line.args

            type = line.command[3:]

            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))
        

        elif line.command[:3] == 'mul':
            dst, src1, src2 = line.args

            type = line.command[3:]

            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))


        elif line.command[:3] == 'shl' or line.command[:3] == 'shr':
            dst, src1, src2 = line.args

            type = line.command[3:]
            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))
        
        elif line.command[:4] == 'selp':
            type = line.command[3:]
            if type[:2] != '.f':

                dst, src0, src1, src2 = line.args
                if src0[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src0)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src0 + '_bits', src1, src2))

                dst, src0, src1, src2 = line.args
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src0, src1 + '_bits', src2))

        



        
def translate_f1_to_pred(ptx_shader):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        # if line.instructionClass == InstructionClass.VariableDeclaration:

        #     if line.declarationType == DeclarationType.Register:

        #         if line.variableType == '.f1':
        #             line.buildString(DeclarationType.Register, line.vector, '.pred', line.variableName)
        

        if line.instructionClass == InstructionClass.Functional:

            if line.command.split()[0] == 'ld.global.b1':
                
                dst, ptr = line.args
                declaration, _ = ptx_shader.findDeclaration(dst)

                assert declaration.variableType == '.b1'

                newDeclaration = PTXDecleration()
                newDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
                newDeclaration.buildString(DeclarationType.Register, None, '.u16', declaration.variableName + '_u16')

                newLoad = PTXFunctionalLine()
                newLoad.leadingWhiteSpace = line.leadingWhiteSpace
                newLoad.buildString('ld.global.u16', (dst + '_u16', ptr))

                newAnd = PTXFunctionalLine()
                newAnd.leadingWhiteSpace = line.leadingWhiteSpace
                newAnd.buildString('and.b16', (dst + '_u16', dst + '_u16', '%const1_u16'))

                newSetp = PTXFunctionalLine()
                newSetp.leadingWhiteSpace = line.leadingWhiteSpace
                newSetp.buildString('setp.eq.u16', (dst, dst + '_u16', '%const1_u16'))


                declaration.buildString(declaration.declarationType, declaration.vector, '.pred', declaration.variableName)
                ptx_shader.lines.remove(line)
                ptx_shader.lines[index:index] = (newDeclaration, newLoad, newAnd, newSetp)
            

def add_consts(ptx_shader):
    const1_u16_declaration = PTXDecleration()
    const1_u16_declaration.leadingWhiteSpace = '\t'
    const1_u16_declaration.buildString(DeclarationType.Register, None, '.u16', '%const1_u16')

    const1_u16_mov = PTXFunctionalLine()
    const1_u16_mov.leadingWhiteSpace = '\t'
    const1_u16_mov.buildString('mov.u16', ('%const1_u16', '1'))

    ptx_shader.addToStart((const1_u16_declaration, const1_u16_mov, PTXLine('\n')))



    const0_u32_declaration = PTXDecleration()
    const0_u32_declaration.leadingWhiteSpace = '\t'
    const0_u32_declaration.buildString(DeclarationType.Register, None, '.u32', '%const0_u32')

    const0_u32_Mov = PTXFunctionalLine()
    const0_u32_Mov.leadingWhiteSpace = '\t'
    const0_u32_Mov.buildString('mov.u32', ('%const0_u32', '0'))

    ptx_shader.addToStart((const0_u32_declaration, const0_u32_Mov, PTXLine('\n')))



    const0_f32_declaration = PTXDecleration()
    const0_f32_declaration.leadingWhiteSpace = '\t'
    const0_f32_declaration.buildString(DeclarationType.Register, None, '.f32', '%const0_f32')

    const0_f32_mov = PTXFunctionalLine()
    const0_f32_mov.leadingWhiteSpace = '\t'
    const0_f32_mov.buildString('mov.f32', ('%const0_f32', '0F00000000'))

    ptx_shader.addToStart((const0_f32_declaration, const0_f32_mov, PTXLine('\n')))


    const1_f32_declaration = PTXDecleration()
    const1_f32_declaration.leadingWhiteSpace = '\t'
    const1_f32_declaration.buildString(DeclarationType.Register, None, '.f32', '%const1_f32')

    const1_f32_mov = PTXFunctionalLine()
    const1_f32_mov.leadingWhiteSpace = '\t'
    const1_f32_mov.buildString('mov.f32', ('%const1_f32', '0F3f800000'))

    ptx_shader.addToStart((const1_f32_declaration, const1_f32_mov, PTXLine('\n')))


def add_temps(ptx_shader):
    temp_pred_declaration = PTXDecleration()
    temp_pred_declaration.leadingWhiteSpace = '\t'
    temp_pred_declaration.buildString(DeclarationType.Register, None, '.pred', '%temp_pred')

    ptx_shader.addToStart((temp_pred_declaration, ))


    temp_f32_declaration = PTXDecleration()
    temp_f32_declaration.leadingWhiteSpace = '\t'
    temp_f32_declaration.buildString(DeclarationType.Register, None, '.f32', '%temp_f32')

    ptx_shader.addToStart((temp_f32_declaration, ))


    temp_u32_declaration = PTXDecleration()
    temp_u32_declaration.leadingWhiteSpace = '\t'
    temp_u32_declaration.buildString(DeclarationType.Register, None, '.u32', '%temp_u32')

    ptx_shader.addToStart((temp_u32_declaration, ))


    temp_u64_declaration = PTXDecleration()
    temp_u64_declaration.leadingWhiteSpace = '\t'
    temp_u64_declaration.buildString(DeclarationType.Register, None, '.u64', '%temp_u64')

    ptx_shader.addToStart((temp_u64_declaration, ))


    ptx_shader.addToStart((PTXLine('\n'), ))



def translate_ALU(ptx_shader):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.fpow:
            dst, src1, src2 = line.args # dst = src1 ^ src2 ?

            declaration, _ = ptx_shader.findDeclaration(dst)
            assert declaration.variableType == '.f32'

            logLine = PTXFunctionalLine()
            logLine.leadingWhiteSpace = line.leadingWhiteSpace
            logLine.buildString('lg2.approx.f32', (dst, src1))

            mulLine = PTXFunctionalLine()
            mulLine.leadingWhiteSpace = line.leadingWhiteSpace
            mulLine.buildString('mul.f32', (dst, dst, src2))

            expLine = PTXFunctionalLine()
            expLine.leadingWhiteSpace = line.leadingWhiteSpace
            expLine.buildString('ex2.approx.f32', (dst, dst))

            ptx_shader.lines.remove(line)
            ptx_shader.lines[index:index] = (logLine, mulLine, expLine)
        

        elif line.functionalType == FunctionalType.flrp:
            dst, src0, src1, src2 = line.args # dst = src0 * (1 - src2) + src1 * src2

            sub = PTXFunctionalLine()
            sub.leadingWhiteSpace = line.leadingWhiteSpace
            sub.buildString('sub.f32', (dst, '%const1_f32', src2))

            mul0 = PTXFunctionalLine()
            mul0.leadingWhiteSpace = line.leadingWhiteSpace
            mul0.buildString('mul.f32', (dst, src0, dst))

            mul1 = PTXFunctionalLine()
            mul1.leadingWhiteSpace = line.leadingWhiteSpace
            mul1.buildString('mul.f32', ('%temp_f32', src2, src1))

            add = PTXFunctionalLine()
            add.leadingWhiteSpace = line.leadingWhiteSpace
            add.comment = line.comment
            add.buildString('add.f32', (dst, dst, '%temp_f32'))

            ptx_shader.lines[index:index + 1] = (sub, mul0, mul1, add)
        

        elif line.functionalType == FunctionalType.bcsel:
            dst, src0, src1, src2 = line.args

            src1Declaration, _ = ptx_shader.findDeclaration(src1)
            src2Declaration, _ = ptx_shader.findDeclaration(src2)
            if not src1Declaration.isLoadConst:
                type = src1Declaration.variableType
            elif not src2Declaration.isLoadConst:
                type = src2Declaration.variableType
            else:
                type = '.f32'
            
            dstDeclaration, _ = ptx_shader.findDeclaration(dst)
            dstDeclaration.buildString(dstDeclaration.declarationType, dstDeclaration.vector, type, dstDeclaration.variableName)

            line.buildString('selp' + type, (dst, src1, src2, src0))

            # line.buildString('selp.f32', (dst, src1, src2, src0))
        
        elif line.functionalType == FunctionalType.pack_64_2x32_split:
            dst, src0, src1 = line.args

            src0Declaration, _ = ptx_shader.findDeclaration(src0)
            assert src0Declaration.variableType == '.u32'

            cvt1 = PTXFunctionalLine()
            cvt1.leadingWhiteSpace = line.leadingWhiteSpace
            cvt1.buildString('cvt.u64.u32', ('%temp_u64', src1))

            shl = PTXFunctionalLine()
            shl.leadingWhiteSpace = line.leadingWhiteSpace
            shl.buildString('shl.b64', (dst, '%temp_u64', src1))

            cvt0 = PTXFunctionalLine()
            cvt0.leadingWhiteSpace = line.leadingWhiteSpace
            cvt0.buildString('cvt.u64.u32', ('%temp_u64', src0))

            orLine = PTXFunctionalLine()
            orLine.leadingWhiteSpace = line.leadingWhiteSpace
            orLine.comment = line.comment
            orLine.buildString('or.b64', (dst, dst, '%temp_u64'))

            ptx_shader.lines[index:index + 1] = (cvt1, shl, cvt0, orLine)
        
        elif line.functionalType == FunctionalType.b2f32:
            dst, src = line.args
            line.buildString('selp.f32', (dst, '0F3f800000', '0F00000000', src))
        
        elif line.functionalType == FunctionalType.fsign:
            dst, src = line.args

            ldLine = PTXFunctionalLine()
            ldLine.leadingWhiteSpace = line.leadingWhiteSpace
            ldLine.buildString('mov.f32', (dst, '0F3f800000'))

            copysignfLine = PTXFunctionalLine()
            copysignfLine.leadingWhiteSpace = line.leadingWhiteSpace
            copysignfLine.comment = line.comment
            copysignfLine.buildString('copysignf', (dst, src))

            ptx_shader.lines[index:index + 1] = (ldLine, copysignfLine)
        

        elif line.functionalType == FunctionalType.fsat:
            dst, src = line.args

            maxLine = PTXFunctionalLine()
            maxLine.leadingWhiteSpace = line.leadingWhiteSpace
            maxLine.buildString('max.f32', (dst, src, '%const0_f32'))

            minLine = PTXFunctionalLine()
            minLine.leadingWhiteSpace = line.leadingWhiteSpace
            minLine.buildString('min.f32', (dst, dst, '%const1_f32'))

            ptx_shader.lines[index:index + 1] = (maxLine, minLine)





def translate_texture_instructions(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.txl:
            dst, texture, sampler, coord, lod = line.args

            newDstNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            newCoordNames, _, _, _ = unwrapp_vector(ptx_shader, coord, coord)
            line.buildString(line.functionalType, [texture, sampler] + newDstNames + newCoordNames[0:2] + [lod, ])
        elif line.functionalType == FunctionalType.tex:
            if len(line.args) == 4:
                dst, texture, sampler, coord = line.args
            elif len(line.args) == 5:
                dst, texture, sampler, coord, dumb = line.args
            
            declaration, _ = ptx_shader.findDeclaration(dst)
            if declaration.isVector():
                newDstNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            else:
                newDstNames = [dst, dst, dst, dst]
            newCoordNames, _, _, _ = unwrapp_vector(ptx_shader, coord, coord)
            if len(newCoordNames) < 3:
                newCoordNames.append(str(0))
                assert(len(newCoordNames) == 3)
            line.buildString(FunctionalType.tex, newDstNames + [texture, sampler] + newCoordNames[0:3])
        elif line.functionalType == FunctionalType.txs:
            dst, texture, lod = line.args

            newDstNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            line.buildString(FunctionalType.txs, newDstNames + [texture, ] + [lod, ])
        elif line.functionalType == FunctionalType.txf:
            dst, texture, coord, lod = line.args
            newCoordNames, _, _, _ = unwrapp_vector(ptx_shader, coord, coord)

            newDstNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            line.buildString(FunctionalType.txs, newDstNames + [texture, ] + newCoordNames + [lod, ])
            


def translate_special_intrinsics(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.shader_clock:
            dst, memory_scope = line.args
            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            line.buildString(FunctionalType.shader_clock, newRegNames[0:2])
        elif line.functionalType == FunctionalType.load_frag_coord:
            dst = line.args[0]
            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            line.buildString(FunctionalType.load_frag_coord, newRegNames)
        
        # if line.functionalType == FunctionalType.report_ray_intersection:
        #     dst, src0, src1 = line.args

        #     # dstDeclaration, dstDeclarationIndex = ptx_shader.findDeclaration(dst)
        #     line.buildString(line.functionalType, ('%temp_u32', src0, src1))

        #     setpLine = PTXFunctionalLine()
        #     setpLine.leadingWhiteSpace = line.leadingWhiteSpace
        #     setpLine.buildString('setp.ne.u32', (dst, '%temp_u32', '%const0_u32'))

        #     ptx_shader.lines[index + 1:index + 1] = (setpLine, )



def add_extra_thread_return(ptx_shader):
    thread_return_code = """.reg .u32 %launch_ID_0;
.reg .u32 %launch_ID_1;
.reg .u32 %launch_ID_2;
load_ray_launch_id %launch_ID_0, %launch_ID_1, %launch_ID_2;

.reg .u32 %launch_Size_0;
.reg .u32 %launch_Size_1;
.reg .u32 %launch_Size_2;
load_ray_launch_size %launch_Size_0, %launch_Size_1, %launch_Size_2;


.reg .pred %bigger_0;
setp.ge.u32 %bigger_0, %launch_ID_0, %launch_Size_0;

.reg .pred %bigger_1;
setp.ge.u32 %bigger_1, %launch_ID_1, %launch_Size_1;

.reg .pred %bigger_2;
setp.ge.u32 %bigger_2, %launch_ID_2, %launch_Size_2;

@%bigger_0 bra shader_exit;
@%bigger_1 bra shader_exit;
@%bigger_2 bra shader_exit;"""

    lines = [PTXLine('\t' + line + '\n') for line in thread_return_code.split("\n")]
    lines.append(PTXLine('\n'))

    ptx_shader.addToStart(lines)








def main():
    unique_ID = 0
    assert len(sys.argv) == 2
    shaderFolder = sys.argv[1]

    shaders = []
    for shaderFile in os.listdir(shaderFolder):
        if 'translated' in shaderFile:
            continue
        shaders.append(PTXShader(os.path.join(shaderFolder, shaderFile)))
    
    shaderIDs = {}
    for shader in shaders:
        if shader.getShaderType() in shaderIDs:
            shaderIDs[shader.getShaderType()].append(shader.getShaderID())
        else:
            shaderIDs[shader.getShaderType()] = [shader.getShaderID(), ]

    
    for shader in shaders:
        add_consts(shader)
        add_temps(shader)

        translate_load_const(shader)
        translate_descriptor_set_instructions(shader)
        translate_deref_instructions(shader)
        translate_trace_ray(shader, shaderIDs)
        translate_decl_var(shader)
        translate_load_GL_instructions(shader)
        translate_image_deref(shader)
        translate_exit(shader)
        translate_texture_instructions(shader)
        translate_special_intrinsics(shader)

        translate_vector_operands(shader, unique_ID)

        translate_ALU(shader)

        translate_phi(shader)
        
        translate_const_operands(shader)

        translate_f1_to_pred(shader)

        if shader.getShaderType() == ShaderType.Ray_generation:
            add_extra_thread_return(shader)
        

        shader.writeToFile()


main()