from synthesis.synthesizer.decider import *
from synthesis.synthesizer.tf_to_torch.torch_result import *
from synthesis.synthesizer.synthesizer import *
from synthesis.search_structure import *
from commons.test_utils import execute_api_call, extract_api_arguments_torch, eval_forward_pass_torch
import numpy as np
import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import concurrent.futures


class Z3Decider(Decider):

    def __init__(self, test_cases: List[TestCase], matching_apis: List[LibraryAPI]):
        super().__init__(test_cases)
        self.matching_apis = matching_apis

    def is_number(self, text):
        if 'zero' in text:
            # For some reason zero is not CD in nltk
            return True
        text = nltk.word_tokenize(text)
        pos = nltk.pos_tag(text)
        for i in range(len(pos)):
            word, pos_tag = pos[i]
            if pos_tag == 'CD':
                return True
        return False

    def preprocess(self, msg):
        sent = nltk.word_tokenize(msg)
        sent = nltk.pos_tag(sent)
        return sent

    def nlp_tagger(self, err_msg):
        msg = self.preprocess(err_msg)
        pattern = 'NP: {<NN>*<IN><JJ>?<NN>}'
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(msg)
        iob_tagged = tree2conlltags(cs)
        return iob_tagged

    def mutate_constraint(self, program, arg, mutated_idx):
        test_program = program.code[0].split('(')
        # 'self.var5 = torch.nn.Conv2d(1,3,0,stride=0,padding=0,dilation=1,groups=1,bias=True,padding_mode=\'zeros\')'

        test_list = (test_program[1].replace(')', '')).split(',')

        for idx, change_arg in enumerate(test_list):
            if change_arg == arg and idx not in mutated_idx:
                test_list[idx] = change_arg.replace('0', '1')
                target_mute = idx
                mutated_idx.append(target_mute)
                break
        # done change list
        mutated_code = test_program[0] + '(' + ','.join(test_list) + ')'
        program.code[0] = mutated_code
        z3res = self.analyze(program)
        return [z3res.error_message(), target_mute]

    def param_not_supported(self, iob_tagged, params, program):
        for idx, word in enumerate(iob_tagged):
            if word[1] == 'JJ':
                target_arg = iob_tagged[idx + 1][0]
                break
        for idx, param in enumerate(params):
            if target_arg in param:
                target_idx = idx + 1
                var = z3.Var(target_idx, IntSort())
                z3_constr = var > 0, [program.argument_vars[idx]]
                # constraint = 'var' + str(target_idx) + '> 0', [program.argument_vars[idx]]
                break
        return z3_constr

    def faulty_matrix_idx(self, iob, params, program):
        arg_constr = ''
        for idx, word in enumerate(iob):
            if 'B-NP' in word[2] and '[' not in word[0]:
                main_arg = word[0]
            if 'NP' in word[2] and 'JJ' in word[1]:
                arg_constr = word[0]
            if '[' in word[0]:
                matrix_start = idx
            if ']' in word[0]:
                matrix_end = idx
        matrix_words = iob[matrix_start:matrix_end]
        # Look for number before ":"
        bad_arg_iob = iob[(matrix_start - 2)]
        if 'N' in bad_arg_iob[1]:
            bad_arg = bad_arg_iob[0]
        arg_count = 0
        for word in matrix_words:
            if ',' in word[0]:
                continue
            else:
                if bad_arg and bad_arg in word:
                    arg_name_idx = arg_count - 1
                    break
                arg_count += 1
        z3_constr = ''

        if main_arg and arg_constr:
            z3_constr = 'param ' + main_arg + ', arg' + str(arg_count) + ' is ' + arg_constr
            if arg_constr and 'neg' in arg_constr:
                target_idx = arg_name_idx + 1
                var = z3.Var(target_idx, IntSort())
                z3_constr = var > 0, [program.argument_vars[arg_name_idx]]
        return z3_constr

    def zero_param(self, err_msg, iob_tagged, program, params):
        constraint = ''
        for idx, word in enumerate(iob_tagged):
            if 'I-NP' in word[2]:
                if idx == (len(iob_tagged) - 1) or 'O' in iob_tagged[idx + 1][2]:
                    constraint = word[0]
        if self.is_number(constraint):
            if 'zero' in constraint:
                constraint = '0'
            arg_count = 1
            mutated_idx = []
            for arg in params:
                if constraint in arg:
                    # TODO this is where I add mutation
                    mutated_msg = self.mutate_constraint(program, arg, mutated_idx)
                    if type(mutated_msg[0]) == list:
                        mutated_msg[0] = str(mutated_msg[0]).replace('[', '').replace(']', '').replace('\'', '')
                    if mutated_msg[0] != err_msg:
                        target_idx = mutated_msg[1] + 1
                        # contr_list = [target_idx, [program.argument_vars[mutated_msg[1]]]]
                        # constraint = 'var' + str(target_idx) + ' > 0', [program.argument_vars[mutated_msg[1]]]
                        var = z3.Var(target_idx, IntSort())
                        z3_constr = var > 0, [program.argument_vars[mutated_msg[1]]]
                        break
                else:
                    arg_count += 1
        return z3_constr

    def error_message_understanding(self, raw_error_message: List[str], program: Program) -> (Constraint, List[str]):
        test_case = self.test_cases[0]
        # map each input arg to arg
        constraint = '', ['']
        code = program.code[0]
        params = code[code.find("(") + 1:code.find(")")].split(',')
        err_msg = str(raw_error_message)
        err_msg = err_msg.replace("[\'", "").replace("\']", "")
        split_msg = err_msg.split(' ')
        if 'Wrong shape ' in str(raw_error_message) and 'vs' in str(raw_error_message):
            return constraint
        elif 'unmatched' in str(raw_error_message):
            return constraint
        iob_tagged = self.nlp_tagger(err_msg)

        # TODO figure this shit out
        if 'out of bounds' in err_msg:
            print('idk this shit')
        #     for idx, word in enumerate(split_msg):
        #         if word == 'index':
        #             constraint = split_msg[idx + 1]
        #             break
        #     var = z3.Var(target_idx, IntSort())
        #     constraint = var > 0, [program.argument_vars[mutated_msg[1]]]
        #     constraint = 'param' + str(constraint) + ' out of bounds'

        elif 'not supported' in err_msg:
            constraint = self.param_not_supported(iob_tagged, params, program)
        elif '[' in err_msg and ']' in err_msg:
            constraint = self.faulty_matrix_idx(iob_tagged, params, program)
        elif 'zero' in err_msg:
            constraint = self.zero_param(err_msg, iob_tagged, program, params)

        var = z3.Var(0, IntSort())
        # return var > 0, ['in_channels']
        return constraint

    def analyze(self, program: Program) -> Result:
        target_call = program.code[0]
        target_name = target_call.split('=')[1].split('(')[0].strip()
        target_api = list(filter(lambda x: x.id == target_name, self.matching_apis))[0]

        try:
            target_args = extract_api_arguments_torch(target_api, target_call)
        except:
            return TorchResult(False)
        print('Evaluating...', target_call)

        # try to create layer
        success, layer = execute_api_call(target_api, target_args)
        if not success:
            return TorchResult(success, layer)  # bad arguments

        # test cases
        for test in self.test_cases:
            success, output = eval_forward_pass_torch(layer, test.input[0])
            if not success:  # runtime error
                return TorchResult(success, output)
            if test.output.shape != output.shape:  # wrong shape
                return TorchResult(False, [f'Wrong shape {test.output.shape} vs {output.shape}'])
            if not np.allclose(test.output, output):  # we don't know why it failed
                return TorchResult(False)

        return TorchResult(True)
