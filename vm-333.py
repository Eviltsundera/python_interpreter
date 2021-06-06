"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.6/Include/frameobject.h#L17

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: tp.Dict[str, tp.Any],
                 frame_globals: tp.Dict[str, tp.Any],
                 frame_locals: tp.Dict[str, tp.Any]) -> None:
        self.offset = 0
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))
        idx = 0
        while idx < len(instructions):
            instruction = instructions[idx]
            self.offset += 2
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if "jump" in instruction.opname.lower() or self.offset != instruction.offset + 2:
                for instruction_jump in instructions:
                    if instruction_jump.offset == self.offset:
                        idx = instructions.index(instruction_jump)
                        break
            else:
                idx += 1
        return self.return_value

    def call_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-CALL_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L3496
        """
        arguments = self.popn(arg)
        f = self.pop()
        self.push(f(*arguments))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2416
        """
        if arg in self.locals:
            val = self.locals[arg]
        elif arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError("name '%s' is not defined" % arg)
        self.push(val)

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_GLOBAL

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2480
        """
        if arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError("name '%s' is not defined" % arg)
        self.push(val)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_CONST

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1346
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-RETURN_VALUE

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1911
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-POP_TOP

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1361
        """
        self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-MAKE_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L3571

        Parse stack:
            https://github.com/python/cpython/blob/3.8/Objects/call.c#L671

        Call function in cpython:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L4950
        """
        CO_VARARGS = 4
        CO_VARKEYWORDS = 8

        ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
        ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
        ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
        ERR_MISSING_POS_ARGS = 'Missing positional arguments'
        ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
        ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

        # extract arguments of consumed stack
        name = self.pop()  # the qualified name of the function (at TOS)  # noqa
        code = self.pop()  # the code associated with the function (at TOS1)

        # number of positional, keywords and others args
        pos_only_arg_count = code.co_posonlyargcount
        kw_only_arg_count = code.co_kwonlyargcount
        arg_count = code.co_argcount

        # free_var_tuple = self.pop() if CO_VARKEYWORDS == arg & CO_VARKEYWORDS else ()
        # annotation_tuple = self.pop() if CO_VARARGS == arg & CO_VARARGS else ()
        kw_only_dict = self.pop() if 0x02 == arg & 0x02 else {}
        pos_only_tuple = self.pop() if 0x01 == arg & 0x01 else ()

        # flags of existing *args and **kwargs
        is_args = CO_VARARGS == CO_VARARGS & code.co_flags
        is_kwargs = CO_VARKEYWORDS == CO_VARKEYWORDS & code.co_flags

        # TODO: use arg to parse function defaults

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount

            # extract args
            pos_args = code.co_varnames[0: pos_only_arg_count]
            usual_args = code.co_varnames[pos_only_arg_count: arg_count]
            kw_args = code.co_varnames[arg_count: arg_count + kw_only_arg_count]
            defaults_args = dict(zip(reversed(pos_args + usual_args), reversed(pos_only_tuple)))  # TODO MAYBE

            parsed_args: tp.Dict[str, tp.Any] = {}

            args_ind = kw_only_arg_count + arg_count

            if not is_kwargs:
                for arg in pos_args:
                    if arg in kwargs.keys():
                        raise TypeError(ERR_POSONLY_PASSED_AS_KW)

            args_list = list(args)
            for arg in pos_args:
                if len(args_list):
                    parsed_args[arg] = args_list.pop(0)
                elif arg in defaults_args.keys():
                    parsed_args[arg] = defaults_args[arg]
                else:
                    raise TypeError(ERR_MISSING_POS_ARGS)

            for arg in usual_args:
                if len(args_list):
                    parsed_args[arg] = args_list.pop(0)
                    if arg in kwargs.keys():
                        raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                elif arg in kwargs.keys():
                    parsed_args[arg] = kwargs[arg]
                    del kwargs[arg]
                elif arg in defaults_args.keys():
                    parsed_args[arg] = defaults_args[arg]
                else:
                    raise TypeError(ERR_MISSING_POS_ARGS)

            if is_args:
                var_name = code.co_varnames[args_ind]
                if var_name not in parsed_args.keys():
                    parsed_args[var_name] = tuple(args_list)
                elif len(args_list):
                    raise TypeError(ERR_TOO_MANY_POS_ARGS)
            for arg in kw_args:
                if arg not in kwargs.keys():
                    if arg not in kw_only_dict.keys():
                        raise TypeError(ERR_MISSING_KWONLY_ARGS)
                    parsed_args[arg] = kw_only_dict[arg]
                    continue
                parsed_args[arg] = kwargs[arg]
                del kwargs[arg]

            if is_kwargs:
                kwargs_var_name = code.co_varnames[args_ind + 1] if is_args else code.co_varnames[args_ind]
                if kwargs_var_name not in parsed_args.keys():
                    parsed_args[kwargs_var_name] = kwargs
            elif len(kwargs):
                raise TypeError(ERR_TOO_MANY_KW_ARGS)

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-STORE_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2280
        """
        const = self.pop()
        self.locals[arg] = const

    def store_subscr_op(self, arg: str) -> None:
        tos2, tos1, tos0 = self.popn(3)
        tos1[tos0] = tos2
        self.push(tos1[tos0])

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def load_fast_op(self, arg: str) -> None:
        """
            Operation description:
                https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_FAST
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            raise UnboundLocalError("")

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def extended_arg_op(self, count: int) -> None:
        res = 0
        while count:
            count -= 1
            a = self.pop()
            res = (res << 8) | a
        self.push(res)

    def unpack_sequence_op(self, _: tp.Any) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            raise

    # Operators
    # Binary operator

    def binary_matrix_multiply(self) -> None:
        pass

    def binary_add_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a + b)

    inplace_add_op = binary_add_op

    def binary_and_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a & b)

    inplace_and_op = binary_and_op

    def binary_floor_divide_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a // b)

    inplace_floor_divide_op = binary_floor_divide_op

    def binary_lshift_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a << b)

    inplace_lshift_op = binary_lshift_op

    def binary_modulo_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a % b)

    inplace_modulo_op = binary_modulo_op

    def binary_multiply_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a * b)

    inplace_multiply_op = binary_multiply_op

    def binary_or_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a | b)

    inplace_or_op = binary_or_op

    def binary_power_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a ** b)

    inplace_power_op = binary_power_op

    def binary_rshift_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a >> b)

    inplace_rshift_op = binary_rshift_op

    def binary_subscr_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        if not isinstance(a, int):
            self.push(a[b])

    def delete_subscr_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        del a[b]
        self.push(a)

    # nplace_rshift_op = binary_rshift_op

    def binary_subtract_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a - b)

    inplace_subtract_op = binary_subtract_op

    def binary_true_divide_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a / b)

    inplace_true_divide_op = binary_true_divide_op

    def binary_xor_op(self, _: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a ^ b)

    inplace_xor_op = binary_xor_op

    COMPARE_OPERATORS = dict(zip(
        dis.cmp_op,
        [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
            lambda a, b: a in b,
            lambda a, b: a not in b,
            lambda a, b: a is b,
            lambda a, b: a is not b,
            lambda a, b: issubclass(a, Exception) and issubclass(a, b),
        ]))

    # Jumps

    def pop_jump_if_true_op(self, jump: int) -> None:
        if self.pop():
            self.offset = jump

    def pop_jump_if_false_op(self, jump: int) -> None:
        if not self.pop():
            self.offset = jump

    def jump_forward_op(self, off_add: int) -> None:
        self.offset = off_add

    def jump_absolute_op(self, off: int) -> None:
        self.offset = off

    def jump_if_true_or_pop_op(self, off: int) -> None:
        val = self.pop()
        if val:
            self.push(val)
            self.offset = off

    def jump_if_false_or_pop_op(self, off: int) -> None:
        val = self.pop()
        if not val:
            self.push(val)
            self.offset = off

    # Buildings
    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        values = self.popn(count)
        self.push(dict(zip(keys, values)))

    def build_list_unpack_op(self, count: int) -> None:
        self.push([i for arr in self.popn(count) for i in arr])

    def build_map_op(self, count: int) -> None:
        arr = self.popn(2 * count)
        self.push(dict(zip(arr[::2], arr[1::2])))

    def build_map_unpack_op(self, count: int) -> None:
        res = dict()
        for mp in self.popn(count):
            res.update(mp)
        self.push(res)

    def build_map_unpack_with_call_op(self, count: int) -> None:  # same as build_map_unpack_op - may be wrong
        res = dict()
        for mp in self.popn(count):
            res.update(mp)
        self.push(res)

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def build_set_unpack_op(self, count: int) -> None:
        res = set()
        for st in self.popn(count):
            res.update(st)
        self.push(res)

    def build_tuple_unpack_op(self, count: int) -> None:
        res = ()
        for t in self.popn(count):
            res += t
        self.push(res)

    def build_tuple_unpack_with_call_op(self, count: int) -> None:  # same as build_tuple_unpack_op - may be wrong
        res = ()
        for t in self.popn(count):
            res = res + t
        self.push(res)

    def build_string_op(self, count: int) -> None:
        ans = "".join(self.popn(count))
        self.push(ans)

    def build_slice_op(self, count: int) -> None:
        if count == 1:
            f = self.pop()
            self.push(slice(0, f))
        if count == 2:
            b, f = self.popn(count)
            self.push(slice(b, f))
        if count == 3:
            b, f, s = self.popn(count)
            self.push(slice(b, f, s))

    def compare_op_op(self, op: str) -> None:
        a, b = self.popn(2)
        self.push(self.COMPARE_OPERATORS[op](a, b))

    def call_function_ex_op(self, flags: int) -> None:
        var_set = self.pop() if flags else {}
        args_list = self.pop()
        func = self.pop()
        self.push(func(*args_list, **var_set))

    # Unary
    def unary_invert_op(self, _: tp.Any) -> None:
        self.push(~self.pop())

    def unary_negative_op(self, _: tp.Any) -> None:
        self.push(-self.pop())

    def unary_positive_op(self, _: tp.Any) -> None:
        self.push(self.pop())

    def unary_not_op(self, _: tp.Any) -> None:
        self.push(not self.pop())

    def get_iter_op(self, _: tp.Any) -> None:
        self.push(iter(self.pop()))

    def format_value_op(self, _: tp.Any) -> None:
        pass

    # Class
    def load_method_op(self, name: str) -> None:
        obj = self.pop()
        obj_dict = obj.__class__.__dict__
        if name in obj_dict:
            self.push(obj)
            self.push(obj_dict[name])

    def load_build_class_op(self, _: tp.Any) -> None:
        pass

    def call_method_op(self, argc: int) -> None:
        args = self.popn(argc)
        func = self.pop()
        obj = self.pop()
        self.push(func(obj, *args))

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        obj_dict = obj.__class__.__dict__
        if name in obj_dict:
            delattr(obj, name)

    def delete_name_op(self, name: str) -> None:
        del self.locals[name]

    def delete_global_op(self, name: str) -> None:
        del self.globals[name]

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def call_function_kw_op(self, argc: int) -> None:
        kw_args = self.pop()
        args_list = self.popn(argc)
        ind = 0
        kw_dict = dict()
        while ind < len(kw_args):
            kw_dict.update({kw_args[-ind - 1]: args_list[-ind - 1]})
            ind += 1
        func = self.pop()
        if ind:
            self.push(func(*args_list[:-ind], **kw_dict))
        else:
            self.push(func(*args_list))

    def dup_top_op(self) -> None:  # Not working
        top = self.pop()
        self.push(top)
        self.push(top)

    def dup_top_two_op(self, _: tp.Any) -> None:  # Not working
        top1 = self.pop()
        top2 = self.pop()
        self.push(top2)
        self.push(top1)
        self.push(top2)
        self.push(top1)

    def rot_two_op(self, _: tp.Any) -> None:
        top1 = self.pop()
        top2 = self.pop()
        self.push(top1)
        self.push(top2)

    def rot_three_op(self, _: tp.Any) -> None:
        top1 = self.pop()
        top2 = self.pop()
        top3 = self.pop()
        self.push(top1)
        self.push(top3)
        self.push(top2)

    def rot_four_op(self, _: tp.Any) -> None:
        top1 = self.pop()
        top2 = self.pop()
        top3 = self.pop()
        top4 = self.pop()
        self.push(top1)
        self.push(top4)
        self.push(top3)
        self.push(top2)

    def for_iter_op(self, delta: int) -> None:
        tos = self.top()
        try:
            mean = tos.__next__()
            self.push(mean)
        except StopIteration:
            self.pop()
            self.offset = delta


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_text_or_obj: code for interpreting
        """
        globals_context: tp.Dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()


class VirtualMachineError(Exception):
    """For raising errors in the operation of the VM."""
    pass
