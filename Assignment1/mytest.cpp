#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

int main(){
  // Importing Python Module
  /*
  PyObject * myModuleString = PyString_FromString((char *) "path_finder");
  PyObject * myModule = PyImport_Import(myModuleString);

  // Getting Reference to Python Function
  PyObject * myFunction = PyObject_GetAttrString(myModule, (char *) "generateMap");
  PyObject * args = PyTuple_Pack(1, (char *) "$PRACSYS_PATH/prx_core/launches/maze"); // this should always be path to maze text file

  // Getting Your Result



  // need to return a list of lists and print the list
  */

  int res = system("/usr/bin/python create.py") // example - use system call to call python function

}
