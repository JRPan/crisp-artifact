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

from enum import Enum, auto, EnumMeta
import re

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class InstructionClass(Enum):
    VariableDeclaration = auto()
    EntryPoint = auto()
    Functional = auto()
    Empty = auto()
    UNKNOWN = auto()

class PTXLine:
    def __init__(self, fullLine) -> None:
        if fullLine.count(';') > 1:
            print('ERROR: you can write only one instruction per line')
            exit(-1)
        
        self.fullLine = fullLine
        if '//' in fullLine:
            self.command, self.comment = fullLine.split("//", 1)
        else:
            self.command = fullLine
            self.comment = ''
        self.leadingWhiteSpace = re.match(r"\s*", self.command).group()
        self.command = self.command.strip()
        self.instructionClass = PTXLine.getInstructionClass(fullLine)
        self.condition = ''
    
    def buildString(self):
        if len(self.condition) > 0:
            self.fullLine = self.leadingWhiteSpace + self.condition + ' ' + self.command
        else:
            self.fullLine = self.leadingWhiteSpace + self.command
        if len(self.comment) > 0:
            self.fullLine += '; //' + self.comment
        else:
            self.fullLine += ';\n'
    
    def addComment(self, comment):
        self.comment += '//' + comment + '\n'
        self.buildString()

    @staticmethod
    def getInstructionClass(line):
        if len(line) == 0 or line.isspace():
            return InstructionClass.Empty
        
        #print("-%s-%s" % (line, line.isspace()))
        firstWord = line.split(None, 1)[0]
        if firstWord == '.entry':
            return InstructionClass.EntryPoint
        
        if firstWord == '.reg' or firstWord == '.local':
            return InstructionClass.VariableDeclaration
        
        if firstWord in FunctionalType or firstWord.split('.')[0] in FunctionalType:
            return InstructionClass.Functional
        
        if firstWord[0] == '.':
            return InstructionClass.UNKNOWN
        
        if firstWord[0:2] == '//':
            return InstructionClass.Empty
        
        return InstructionClass.Functional
    
    @staticmethod
    def createNewLine(line):
        lineClass = PTXLine.getInstructionClass(line)
        if lineClass == InstructionClass.VariableDeclaration:
            return PTXDecleration(line)
        elif lineClass == InstructionClass.EntryPoint:
            return PTXEntryPoint(line)
        elif lineClass == InstructionClass.Functional:
            return PTXFunctionalLine(line)
        else:
            return PTXLine(line)


class DeclarationType(Enum):
    Register = '.reg'
    Local = '.local'
    Other = auto()


class PTXDecleration (PTXLine):
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)
        if self.command != '':
            self.parse(self.command)
    
    def parse(self, command):
        # print('###')
        # print(command)
        # print('###')
        self.command = command
        firstWord = command.split(None, 1)[0]
        if firstWord == '.reg':
            self.declarationType = DeclarationType.Register
        elif firstWord == '.local':
            self.declarationType = DeclarationType.Local
        else:
            raise NotImplementedError
        
        args = command.split()
        for i in range(0,len(args)):
            if '.field0' in args[i]:
                args[i] = args[i].replace('.','')
            if '@' in args[i]:
                    args[i] = args[i].replace('@','')
            if '-' in args[i]:
                    args[i] = args[i].replace('-','_')
        if '.v' in args[1]:
            self.vector = args[1]
            index = 2
        else:
            self.vector = ''
            index = 1
        
        self.variableType = args[index]
        self.variableName = args[index + 1]
        if self.variableName[-1] == ';':
            self.variableName = self.variableName[:-1]
        
        if index + 2 < len(args):
            args = args[index + 2:]
        else:
            args = None
        
        self.pointerVariableType = None
        self.isLoadConst = False
    
    def buildString(self, declarationType, vector, variableType, variableName):
        assert declarationType != DeclarationType.Other
        self.instructionClass = InstructionClass.VariableDeclaration

        self.declarationType = declarationType
        self.vector = vector
        self.variableType = variableType
        self.variableName = variableName

        if self.vector == None:
            vector = ''
        else:
            vector = ' ' + self.vector
        self.command = '%s%s %s %s' % (declarationType.value, vector, variableType, variableName)

        self.pointerVariableType = None
        self.isLoadConst = False

        super().buildString()
    
    def isVector(self):
        if self.vector is None or self.vector == '':
            return False
        else:
            return True
    
    def vectorSize(self):
        assert self.isVector()
        assert self.vector[0:2] == '.v'
        return int(self.vector[2:])
    
    def bitCount(self):
        return int(self.variableType[2:])


class FunctionalType(Enum, metaclass=MetaEnum):
    load_ray_launch_id = 'load_ray_launch_id'
    load_ray_launch_size = 'load_ray_launch_size'
    vulkan_resource_index = 'vulkan_resource_index'
    load_vulkan_descriptor = 'load_vulkan_descriptor'
    deref_cast = 'deref_cast'
    deref_struct = 'deref_struct'
    deref_array = 'deref_array'
    load_deref = 'load_deref'
    store_deref = 'store_deref'
    trace_ray = 'trace_ray'
    call_miss_shader = 'call_miss_shader'
    call_closest_hit_shader = 'call_closest_hit_shader'
    call_intersection_shader = 'call_intersection_shader'
    alloca = 'alloca'
    decl_var = 'decl_var'
    deref_var = 'deref_var'
    mov = 'mov'
    image_deref_store = 'image_deref_store'
    image_deref_load = 'image_deref_load'
    exit = 'exit'
    ret = 'ret'
    phi = 'phi'
    load_const = 'load_const'
    load_ray_world_to_object = 'load_ray_world_to_object'
    load_ray_object_to_world = 'load_ray_object_to_world'
    load_ray_world_direction = 'load_ray_world_direction'
    load_ray_world_origin = 'load_ray_world_origin'
    fpow = 'fpow'
    flrp = 'flrp'
    bra = 'bra'
    end_trace_ray = 'end_trace_ray'
    bcsel = 'bcsel'
    selp = 'selp'
    pack_64_2x32_split = 'pack_64_2x32_split'
    txl = 'txl'
    tex = 'tex'
    txs = 'txs'
    txf = 'txf'
    b2f32 = 'b2f32'
    fsign = 'fsign'
    shader_clock = 'shader_clock'
    load_frag_coord = 'load_frag_coord'
    report_ray_intersection = 'report_ray_intersection'
    fsat = 'fsat'
    # get_intersection_index = 'get_intersection_index'
    run_intersection = 'run_intersection'
    intersection_exit = 'intersection_exit'
    hit_geometry = 'hit_geometry'
    get_warp_hitgroup = 'get_warp_hitgroup'
    get_hitgroup = 'get_hitgroup'
    get_closest_hit_shaderID = 'get_closest_hit_shaderID'
    get_intersection_shaderID = 'get_intersection_shaderID'
    get_intersection_shader_data_address = 'get_intersection_shader_data_address'
    Other = auto()

class PTXFunctionalLine (PTXLine): # come up with a better name. I mean a line that does sth like mov (eg it's not decleration)
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)
        if self.command != '':
            self.parse(self.command)
    
    def parse(self, command):
        self.command = command

        firstWord = command.split(None, 1)[0]
        if firstWord[-1] == ';':
            firstWord = firstWord[:-1]

        if firstWord in FunctionalType:
            self.functionalType = FunctionalType[firstWord]
            self.fullFunction = self.functionalType
        elif firstWord.split('.')[0] in FunctionalType:
            dotSplit = firstWord.split('.')
            self.functionalType = FunctionalType[dotSplit[0]]
            self.fullFunction = firstWord
            self.vector = None
            for specifier in dotSplit[1:]:
                if specifier[0] == 'v':
                    self.vector = '.' + specifier
                else:
                    self.variableType = '.' + specifier

        else:
            self.functionalType = FunctionalType.Other
            self.fullFunction = firstWord
        

        if len(command.split(None, 1)) > 1:
            args = command.split(None, 1)[1]
            if args[-1] == ';':
                args = args[:-1]
            elif args[-1][-1] == ';':
                args[-1] = args[-1][:-1]
            args = args.split(',')
            for i in range(0,len(args)):
                if '.field0' in args[i]:
                    args[i] = args[i].replace('.','')
                if '@' in args[i]:
                    args[i] = args[i].replace('@','')
                if '-' in args[i]:
                    args[i] = args[i].replace('-','_')
            self.args = [arg.strip() for arg in args]
        else:
            self.args = []
    
    def buildString(self, function=None, args=None):
        self.instructionClass = InstructionClass.Functional
        if function is None:
            function = self.functionalType
        if args is None:
            args = self.args
        
        if isinstance(function, FunctionalType):
            self.command = function.name
            self.functionalType = function
        else:
            self.command = function
            self.functionalType = FunctionalType.Other
        
        self.args = args
        self.command += ' ' + ', '.join(args)
        if self.functionalType == FunctionalType.tex:
            self.command = self.command.replace('tex','tex.2d.v4.f32.f32')
        super().buildString()
    

    # def is_load_const(self):
    #     if self.functionalType != FunctionalType.mov:
    #         return False
    #     if len(self.args) != 2:
    #         return False
    #     if self.args[0][0] != '%':
    #         return False
    #     if self.args[1][0] == '%':
    #         return False
    #     return True


class PTXEntryPoint (PTXLine):
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)


class ShaderType(Enum):
    Ray_generation = auto()
    Closest_hit = auto()
    Miss = auto()
    Intersection = auto()
    Any_hit = auto()
    Callable = auto()
    Vertex = auto()
    Fragment= auto()


class PTXShader:
    def __init__(self, filePath) -> None:
        self.filePath = filePath
        f = open(filePath, "r")
        lineNO = 1
        self.lines = []
        self.vectorVariables = list()
        for line in f:
            print('parsing line %s: %s' % (lineNO, line))
            ptxLine = PTXLine.createNewLine(line)
            if ptxLine.instructionClass == InstructionClass.VariableDeclaration and ptxLine.declarationType == DeclarationType.Register:
                if ptxLine.isVector():
                    self.vectorVariables.append(ptxLine.variableName)
            # print("#1")
            # print(line)
            # print(ptxLine.instructionClass)
            if ptxLine.instructionClass == InstructionClass.Functional:
                print(ptxLine.functionalType)
            # if ptxLine.instructionClass == InstructionClass.Functional:
            #     print(ptxLine.functionalType)
            self.lines.append(ptxLine)
            lineNO += 1
        # exit(-1)
        f.close()
    
    def getShaderType(self):
        for index in range(len(self.lines)):
            line = self.lines[index]
            if line.instructionClass != InstructionClass.EntryPoint:
                continue
            if 'main' not in line.fullLine:
                continue

            if 'RAYGEN' in line.fullLine:
                return ShaderType.Ray_generation
            if 'CLOSEST_HIT' in line.fullLine:
                return ShaderType.Closest_hit
            if 'MISS' in line.fullLine:
                return ShaderType.Miss
            if 'INTERSECTION' in line.fullLine:
                return ShaderType.Intersection
            if 'ANY_HIT' in line.fullLine:
                return ShaderType.Any_hit
            if 'CALLABLE' in line.fullLine:
                return ShaderType.Callable
            if 'VERTEX' in line.fullLine:
                return ShaderType.Vertex
            if 'FRAGMENT' in line.fullLine:
                return ShaderType.Fragment

            
            assert 0
    
    def getShaderID(self):
        for index in range(len(self.lines)):
            line = self.lines[index]
            if line.instructionClass != InstructionClass.EntryPoint:
                continue
            if 'main' not in line.fullLine:
                continue

            func_name = line.fullLine.split()[1]
            assert 'MESA_SHADER_' in func_name and '_main' in func_name
            return int(func_name.split('_')[-2][4:])
            

    
    def findDeclaration(self, name):
        for index in range(len(self.lines)):
            line = self.lines[index]
            if line.instructionClass == InstructionClass.VariableDeclaration:
                if line.variableName == name:
                    return line, index
        return None, None
    
    def writeToFile(self, filePath=None):
        if filePath == None:
            filePath = self.filePath
        f = open(filePath, 'w')
        for line in self.lines:
            f.write(line.fullLine)
        f.close()
    
    def addToEndOfBlock(self, ptxLines, blockName):
        index = 0
        for index in range(len(self.lines)):
            line = self.lines[index]
            if blockName in line.fullLine and 'end_block' in line.fullLine:
                break
        
        if index > 0 and self.lines[index - 1].instructionClass == InstructionClass.Empty:
            index -= 1

        braExists = False
        braIndex = index
        while braIndex >= 0:
            if 'start_block' in self.lines[braIndex].fullLine:
                break
            elif self.lines[braIndex].instructionClass == InstructionClass.Empty:
                braIndex -= 1
            elif self.lines[braIndex].instructionClass == InstructionClass.Functional and self.lines[braIndex].functionalType == FunctionalType.bra:
                braExists = True
                break
            else:
                break
        
        if braExists:
            index = braIndex
        
        self.lines[index:index] = ptxLines
    

    def addToStartOfBlock(self, ptxLines, blockName):
        index = 0
        for index in range(len(self.lines)):
            line = self.lines[index]
            if blockName in line.fullLine and 'start_block' in line.fullLine:
                break
        
        while index < len(self.lines) - 1 and self.lines[index + 1].instructionClass == InstructionClass.Empty:
            index += 1
        
        self.lines[index:index] = ptxLines
    

    def addToStart(self, ptxLines):
        for index in range(len(self.lines)):
            line = self.lines[index]
            if line.instructionClass != InstructionClass.EntryPoint:
                continue
            if 'main' not in line.fullLine:
                continue
            
            index += 1
            self.lines[index:index] = ptxLines

        # index = 0
        # for index in range(len(self.lines)):
        #     line = self.lines[index]
        #     if 'start_block' in line.fullLine:
        #         break
        
        # while index < len(self.lines) - 1 and self.lines[index + 1].instructionClass == InstructionClass.Empty:
        #     index += 1
        
        # self.lines[index:index] = ptxLines