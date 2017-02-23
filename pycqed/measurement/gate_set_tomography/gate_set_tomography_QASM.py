

def gatestring_list_to_QASM(gatestring_list):
    """
    Takes a gatestring list and returns our own peculiar dialect of QASM
    """

    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'T1.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, cl in enumerate(clocks):
        qasm_file.writelines('\ninit_all\n')
