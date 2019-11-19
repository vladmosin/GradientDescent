import sys

from drawing import *
from methods import *
from oracles import *


class Method(Enum):
    GRADIENT_DESCENT = 0,
    NEWTON = 1


class Distribution(Enum):
    UNIFORM = 1,
    GAUSSIAN = 2


class ParsedData:
    def __init__(self, method=None, path=None,
                 search_type=None, epsilon=None,
                 seed=None, distribution=None):
        self.method = method
        self.path = path
        self.search_type = search_type
        self.epsilon = epsilon
        self.seed = seed
        self.distribution = distribution


class ResultData:
    def __init__(self, initial_point=None,
                 optimal_point=None, function_value=None,
                 gradient_value=None, oracle_calls=None,
                 r_k=None, working_time=None,
                 float_operations=None):
        self.optimal_point = optimal_point
        self.initial_point = initial_point
        self.function_value = function_value
        self.gradient_value = gradient_value
        self.oracle_calls = oracle_calls
        self.r_k = r_k
        self.working_time = working_time
        self.float_operations = float_operations

class CGPolicies(Enum):
    CONST = 0,
    SQRT_GRAD_NORM = 1,
    GRAD_NORM = 2


def parse_sysargs(sys_args):  # string
    args = sys_args.split(' ')
    val_by_arg = {}

    for arg in args:
        if len(arg) != 0:
            arg_name, val = arg.split('=')
            if arg_name == '--ds_path':
                val_by_arg['path'] = val
            elif arg_name == '--optimize_method':
                if val == 'gradient':
                    val_by_arg['method'] = Method.GRADIENT_DESCENT
                else:
                    val_by_arg['method'] = Method.NEWTON
            elif arg_name == '--line_search':
                search_methods = {
                    'golden_search': SearchType.TERNARY,
                    'brent': SearchType.BRENT,
                    'armijo': SearchType.ARMIJO,
                    'wolfe': SearchType.WOLFE,
                    'lipschitz': SearchType.LIPSCHITZ
                }

                val_by_arg['line_search'] = search_methods[val]
            elif arg_name == '--seed':
                val_by_arg['seed'] = int(val)
            elif arg_name == '--eps':
                val_by_arg['eps'] = np.float64(val)
            elif arg_name == '--point_distribution':
                if val == 'uniform':
                    val_by_arg['distribution'] = Distribution.UNIFORM
                else:
                    val_by_arg['distribution'] = Distribution.GAUSSIAN
            elif arg_name == '--cg-tolerance-policy':
                policies = {
                    'const': CGPolicies.CONST,
                    'sqrtGradNorm': CGPolicies.SQRT_GRAD_NORM,
                    'gradNorm': CGPolicies.GRAD_NORM
                }
                val_by_arg['policy'] = policies[val]
            elif arg_name == '--cg-tolerance-eta':
                val_by_arg['eta'] = np.float64(val)

    return val_by_arg


def print_result(result_data: ResultData):
    s = '{\n' + \
        '\t \'initial_point\': ' + '\'' + str(result_data.initial_point) + '\',\n' + \
        '\t \'optimal_point\': ' + '\'' + str(result_data.optimal_point) + '\',\n' + \
        '\t \'function_value\': ' + '\'' + str(result_data.function_value) + '\',\n' + \
        '\t \'gradient_value\': ' + '\'' + str(result_data.gradient_value) + '\',\n' + \
        '\t \'float_operations\': ' + '\'' + str(result_data.float_operations) + '\',\n' + \
        '\t \'optimal_point\': ' + '\'' + str(result_data.optimal_point) + '\',\n' + \
        '\t \'oracle_calls\': ' + '\'' + str(result_data.oracle_calls) + '\',\n' + \
        '\t \'r_k\': ' + '\'' + str(result_data.r_k) + '\',\n' + \
        '\t \'working_time\': ' + '\'' + str(result_data.working_time) + '\'\n' + \
        '}'

    print(s)


DIM = 10
ROWS = 1000


def gen_dataset():
    y, X = [], []
    a = np.random.uniform(-1, 1, DIM)
    b = np.random.uniform(-1, 1, 1)[0]
    for _ in range(ROWS):
        x = np.random.normal(0, 1, DIM)
        x[0] = 1
        X.append(x)
        if x.dot(a) >= 0:
            y.append(np.float64(1))
        else:
            y.append(np.float64(0))

    return np.array(y), np.array(X)


def read_dataset(path):
    if path == 'generate_dataset':
        return gen_dataset()
    X, Y = [], []
    feature_values = []
    feature_number = 0

    for line in open(path).readlines():
        feature_value = {}
        elements = line.split(' ')
        y, x = elements[0], elements[1:]

        for elem in x:
            index, value = elem.split(':')
            feature_number = max(int(index), feature_number)
            feature_value[int(index)] = np.float64(value)

        feature_values.append(feature_value)
        Y.append(y)

    feature_number += 1

    for feature_value in feature_values:
        row = [np.float64(0.0)] * feature_number
        for feature in feature_value:
            row[feature] = feature_value[feature]

        X.append(row)

    Y = y_to_01(Y)
    return np.array(Y), np.array(X)


def y_to_01(y):
    uniq_y = np.unique(y)
    return list(map(lambda y: np.float64(0) if y == uniq_y[0] else np.float64(1), y))


def gen_initial_point(distribution, dimension):
    if distribution == Distribution.UNIFORM:
        return np.random.uniform(-1, 1, dimension)
    else:
        return np.random.normal(0, 10 ** 0.5, dimension)


def get_eps_lambda(policy, eta):
    lambdas = {
        CGPolicies.CONST: lambda _: eta,
        CGPolicies.GRAD_NORM: lambda x: x * eta,
        CGPolicies.SQRT_GRAD_NORM: lambda x: x ** 0.5 * eta
    }

    return lambdas[policy]


def main(draw=False):
    result_data = ResultData()
    val_by_arg = parse_sysargs(' '.join(sys.argv[1:]))
    #val_by_arg = parse_sysargs(args)

    y, X = read_dataset(val_by_arg['path'])
    f_holder = FunctionHolder(X, y)

    initial_point = gen_initial_point(val_by_arg['distribution'], X.shape[1])

    result_data.initial_point = np.copy(initial_point)

    start_time = time.time()

    if val_by_arg['method'] == Method.GRADIENT_DESCENT:
        point, statistics = gradient_descent(f_holder, initial_point, search_type=val_by_arg['line_search'],
                                             stop_criterion=val_by_arg['eps'])
    else:
        if 'policy' in val_by_arg and 'eta' in val_by_arg:
            policy = val_by_arg['policy']
            eta = val_by_arg['eta']
            point, statistics = newton(f_holder, initial_point / 100,
                                       stop_criterion=val_by_arg['eps'], eps=get_eps_lambda(policy, eta))
        else:
            point, statistics = newton(f_holder, initial_point / 100, stop_criterion=val_by_arg['eps'])

    result_data.working_time = (time.time() - start_time)
    result_data.optimal_point = point
    result_data.oracle_calls = statistics.oracles[-1]
    result_data.float_operations = statistics.operations[-1]
    result_data.function_value = f_holder.f(point)
    result_data.gradient_value = f_holder.grad(point)
    result_data.r_k = (norm(f_holder.grad(point)) / norm(f_holder.grad(initial_point))) ** 2

    if draw:
        return point, statistics, f_holder
    else:
        print_result(result_data)


if __name__ == '__main__':
    main()
