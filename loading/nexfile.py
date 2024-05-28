import numpy as np
import nex.nexfile
import os

class Nex:
    """A basic class to allow Nex/Nex5 data unwrapping in Python"""

    def __init__(self, filename, path=os.getcwd()):
        self.filename = filename
        self.path = path
        file = self.load_file()
        self.dict = {v['Header']['Name']: {p: v[p] for p in list(v.keys())[1:]} for v in file['Variables']}
        self.headers = {v['Header']['Name']: v['Header'] for v in file['Variables']}
        self.main_header = file['FileHeader']
        try:
            self.metadata = file['MetaData']
        except KeyError:
            self.metadata = None

    def __repr__(self):
        infos = f"File: {self.filename}\n"
        for k, var in enumerate(self.keys()):
            infos += f"{var} "
            infos += (np.max([len(v) for v in self.keys()]) - len(str(var))) * ' '
            infos += ':'
            infos += ', '.join(key for key in self.dict[var].keys())
            infos += "\n"
        return infos

    def load_file(self):
        reader = nex.nexfile.Reader()
        try:
            return reader.ReadNexFile(self.path + self.filename + '.nex')
        except FileNotFoundError:
            return reader.ReadNexFile(self.path + self.filename + '.nex5')

    def keys(self):
        return self.dict.keys()

    def get(self, names, vartype=None):
        """
        Gets any variable of the file given it's name or a list of possible names.
         - If a variable type is specified it returns the variable corresponding to the specified type
         - If no variable type is specified and several variable types are available it returns the dictionnary of possible variables
         - If no variable type is specified and only one variable type is avalaible it returns the avalaible variable type
        Lists of np.arrays (e.g Intervals) are automatically stacked
        """
        names = [names] if type(names) is str else names
        variables = [self.dict[n] for n in names if n in self.keys()]

        if not variables:
            raise KeyError(f"In {self.filename}, no variable corresponding to {names}, found {self.keys()}")

        types = [list(var.keys()) for var in variables]

        if vartype is not None:
            found_var = [np.stack(var[vartype]) for var in variables if vartype in var]
            if not found_var:
                raise KeyError(f"No variable of type {vartype} for variable name {names}, found {types[1:]}")
            return found_var[0]

        if len(types) == 1:
            return np.stack(variables[0][types[0][0]])

        return {t: np.stack(var[t]) for var, t in zip(variables, types)}
