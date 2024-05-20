import sys
import traceback


def force_load(name):
    file = open(name + '.py', 'r')
    lines = file.readlines()
    size = len(lines)
    for ind in range(size):
        try:
            exec("".join(lines), globals())
        except SyntaxError as err:
            print(err.lineno)
            if err.lineno is not None:
                lines.pop(err.lineno - 1)
        except Exception as err:
            lines.pop(traceback.extract_tb(sys.exc_info()[2:][0])[-1][1] - 1)
    ldict = {}
    exec("".join(lines), globals(), ldict)
    file.close()
    return ldict
