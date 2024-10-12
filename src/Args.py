import argparse

class ArgparseTool():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    def NeedBoolean(self,name,abbreviation,helptext, dest,required=True, default=True):
        self.parser.add_argument(abbreviation, name, help=helptext,action="store_true",dest=dest,default = default)
    def NeedText(self,name,abbreviation,helptext, dest, required=True, default=None):
        if required == False and default is None:
            assert IndexError('Without Default')
        self.parser.add_argument(abbreviation,name, help=helptext,type=str, required=required,dest=dest,default = default)
    def GetArgs(self):
        return self.parser.parse_args()
