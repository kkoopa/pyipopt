/*
 * Copyright (c) 2008, Eric You Xu, Washington University All rights
 * reserved. Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. * Redistributions in
 * binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution. * Neither the name of the
 * Washington University nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Added "eval_intermediate_callback" by
 * OpenMDAO at NASA Glenn Research Center, 2010 and 2011
 *
 * Changed logger from code contributed by alanfalloon
*/

#include "hook.h"
#ifndef _WIN32
//#include <unistd.h>
#endif

void logger(const char *fmt, ...)
{
	if (user_log_level == VERBOSE) {
		va_list ap;
		va_start(ap, fmt);
		vprintf(fmt, ap);
		va_end(ap);
		printf("\n");
		fflush(stdout);
	}
}

Bool eval_intermediate_callback(Index alg_mod,	/* 0 is regular, 1 is resto */
				Index iter_count, Number obj_value,
				Number inf_pr, Number inf_du,
				Number mu, Number d_norm,
				Number regularization_size,
				Number alpha_du, Number alpha_pr,
				Index ls_trials, UserDataPtr data)
{
	DispatchData *myowndata;
	UserDataPtr user_data;

	long result_as_long;
	Bool result_as_bool;

	PyObject *python_algmod;
	PyObject *python_iter_count;
	PyObject *python_obj_value;
	PyObject *python_inf_pr;
	PyObject *python_inf_du;
	PyObject *python_mu;
	PyObject *python_d_norm;
	PyObject *python_regularization_size;
	PyObject *python_alpha_du;
	PyObject *python_alpha_pr;
	PyObject *python_ls_trials;

	PyObject *arglist = NULL;

	PyObject *result;

	logger("[Callback:E]intermediate_callback");

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	python_algmod = Py_BuildValue("i", alg_mod);
	python_iter_count = Py_BuildValue("i", iter_count);
	python_obj_value = Py_BuildValue("d", obj_value);
	python_inf_pr = Py_BuildValue("d", inf_pr);
	python_inf_du = Py_BuildValue("d", inf_du);
	python_mu = Py_BuildValue("d", mu);
	python_d_norm = Py_BuildValue("d", d_norm);
	python_regularization_size =
	    Py_BuildValue("d", regularization_size);
	python_alpha_du = Py_BuildValue("d", alpha_du);
	python_alpha_pr = Py_BuildValue("d", alpha_pr);
	python_ls_trials = Py_BuildValue("i", ls_trials);

	if (user_data != NULL)
		arglist = Py_BuildValue("(OOOOOOOOOOOO)",
					python_algmod,
					python_iter_count,
					python_obj_value,
					python_inf_pr,
					python_inf_du,
					python_mu,
					python_d_norm,
					python_regularization_size,
					python_alpha_du,
					python_alpha_pr,
					python_ls_trials,
					(PyObject *) user_data);
	else
		arglist = Py_BuildValue("(OOOOOOOOOOO)",
					python_algmod,
					python_iter_count,
					python_obj_value,
					python_inf_pr,
					python_inf_du,
					python_mu,
					python_d_norm,
					python_regularization_size,
					python_alpha_du,
					python_alpha_pr, python_ls_trials);

	result =
	    PyObject_CallObject(myowndata->eval_intermediate_callback_python,
				arglist);

	if (!result)
		PyErr_Print();

	result_as_long = PyLong_AsLong(result);
	result_as_bool = (Bool) result_as_long;

	Py_DECREF(result);
	Py_CLEAR(arglist);
	logger("[Callback:R] intermediate_callback");
	return result_as_bool;
}

Bool
eval_f(Index n, Number * x, Bool new_x, Number * obj_value, UserDataPtr data)
{
	npy_intp dims[1];
	DispatchData *myowndata;
	UserDataPtr user_data;

	PyObject *arrayx;

	PyObject *arglist;
	PyObject *result;

	logger("[Callback:E] eval_f");

	dims[0] = n;

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	// import_array ();

	import_array1(FALSE);
	arrayx =
	    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char *)x);
	if (!arrayx)
		return FALSE;

	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject *arg1;
		PyObject *tempresult;
		arg1 = Py_BuildValue("(O)", arrayx);
		tempresult = PyObject_CallObject(
        myowndata->apply_new_python, arg1);
		if (tempresult == NULL) {
			logger("[Error] Python function apply_new returns NULL");
      PyErr_Print();
			Py_DECREF(arg1);
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}

	if (user_data != NULL) {
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject *) user_data);
  } else {
		arglist = Py_BuildValue("(O)", arrayx);
  }

	result = PyObject_CallObject(myowndata->eval_f_python, arglist);

	if (result == NULL) {
    logger("[Error] Python function eval_f returns NULL");
		PyErr_Print();
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
	}

	*obj_value = PyFloat_AsDouble(result);

  if (PyErr_Occurred()) {
    logger("[Error] Python function eval_f returns non-PyFloat");
		PyErr_Print();
		Py_DECREF(result);
		Py_DECREF(arrayx);
		Py_CLEAR(arglist);
		return FALSE;
  }

	Py_DECREF(result);
	Py_DECREF(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_f");
	return TRUE;
}

Bool
eval_grad_f(Index n, Number * x, Bool new_x, Number * grad_f, UserDataPtr data)
{
	DispatchData *myowndata;
	UserDataPtr user_data;

	npy_intp dims[1];

	PyObject *arrayx;
	PyObject *arglist;

	double *tempdata;
	PyArrayObject *result;

	int i;

	logger("[Callback:E] eval_grad_f");

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	if (myowndata->eval_grad_f_python == NULL)
		PyErr_Print();

	/* int dims[1]; */
	dims[0] = n;
	// import_array ();

	import_array1(FALSE);

	/*
	 * PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE
	 * , (char*) x);
	 */
	arrayx =
	    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char *)x);
	if (!arrayx)
		return FALSE;

	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject *arg1;
		PyObject *tempresult;
		arg1 = Py_BuildValue("(O)", arrayx);
		tempresult = PyObject_CallObject(
        myowndata->apply_new_python, arg1);
		if (tempresult == NULL) {
			logger("[Error] Python function apply_new returns NULL");
      PyErr_Print();
			Py_DECREF(arg1);
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}

	if (user_data != NULL)
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject *) user_data);
	else
		arglist = Py_BuildValue("(O)", arrayx);

	result = (PyArrayObject *) PyObject_CallObject(
      myowndata->eval_grad_f_python, arglist);

	if (result == NULL) {
    logger("[Error] Python function eval_grad_f returns NULL");
		PyErr_Print();
    return FALSE;
  }

  if (!PyArray_Check(result)) {
    logger("[Error] Python function eval_grad_f returns non-PyArray");
    Py_DECREF(result);
    return FALSE;
  }

	tempdata = (double *) PyArray_DATA(result);
	for (i = 0; i < n; i++)
		grad_f[i] = tempdata[i];

	Py_DECREF(result);
	Py_CLEAR(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_grad_f");
	return TRUE;
}

Bool
eval_g(Index n, Number * x, Bool new_x, Index m, Number * g, UserDataPtr data)
{

	DispatchData *myowndata;
	UserDataPtr user_data;

	npy_intp dims[1] = {n};
	int i;
	double *tempdata;

	PyObject *arrayx;

	PyObject *arglist;
	PyArrayObject *result;

	logger("[Callback:E] eval_g");

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	if (myowndata->eval_g_python == NULL)
		PyErr_Print();

	// import_array ();
	import_array1(FALSE);

	/*
	 * PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE
	 * , (char*) x);
	 */
	arrayx =
	    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char *)x);
	if (!arrayx)
		return FALSE;

	if (new_x && myowndata->apply_new_python) {
		/* Call the python function to applynew */
		PyObject *arg1;
		PyObject *tempresult;

		arg1 = Py_BuildValue("(O)", arrayx);
		tempresult = PyObject_CallObject(
        myowndata->apply_new_python, arg1);
		if (tempresult == NULL) {
			logger("[Error] Python function apply_new returns NULL");
      PyErr_Print();
			Py_DECREF(arg1);
			return FALSE;
		}
		Py_DECREF(arg1);
		Py_DECREF(tempresult);
	}

	if (user_data != NULL)
		arglist = Py_BuildValue("(OO)", arrayx, (PyObject *) user_data);
	else
		arglist = Py_BuildValue("(O)", arrayx);

	result = (PyArrayObject *) PyObject_CallObject(
      myowndata->eval_g_python, arglist);

  if (result == NULL) {
    logger("[Error] Python function eval_g returns NULL");
		PyErr_Print();
    return FALSE;
  }

  if (!PyArray_Check(result)) {
    logger("[Error] Python function eval_g returns non-PyArray");
    Py_DECREF(result);
    return FALSE;
  }

	tempdata = (double *) PyArray_DATA(result);
	for (i = 0; i < m; i++) {
		g[i] = tempdata[i];
	}

	Py_DECREF(result);
	Py_CLEAR(arrayx);
	Py_CLEAR(arglist);
	logger("[Callback:R] eval_g");
	return TRUE;
}

Bool
eval_jac_g(Index n, Number * x, Bool new_x,
	   Index m, Index nele_jac,
	   Index * iRow, Index * jCol, Number * values, UserDataPtr data)
{

	DispatchData *myowndata;
	UserDataPtr user_data;

	int i;
	long *rowd = NULL;
	long *cold = NULL;

	double *tempdata;

	npy_intp dims[1] = {n};

	logger("[Callback:E] eval_jac_g");

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	if (myowndata->eval_grad_f_python == NULL)	/* Why??? */
		PyErr_Print();

	if (values == NULL) {
		PyObject *arrayx;
		PyObject *arglist;
		PyObject *result;
		PyArrayObject *row;
		PyArrayObject *col;

		/* import_array (); */
		import_array1(FALSE);

		arrayx =
		    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
					      (char *)x);
		if (!arrayx)
			return FALSE;

		if (user_data != NULL)
			arglist = Py_BuildValue("(OOO)",
						arrayx, Py_True,
						(PyObject *) user_data);
		else
			arglist = Py_BuildValue("(OO)", arrayx, Py_True);

		result =
		    PyObject_CallObject(myowndata->eval_jac_g_python, arglist);
		if (!result) {

			logger("[PyIPOPT] return from eval_jac_g is null\n");
			/* TODO: need to deal with reference counting here */
			return FALSE;
		}
		if (!PyTuple_Check(result)) {
			PyErr_Print();
		}
		row =
		    (PyArrayObject *) PyTuple_GetItem(result, 0);
		col =
		    (PyArrayObject *) PyTuple_GetItem(result, 1);

		if (!row || !col || !PyArray_Check(row) || !PyArray_Check(col)) {
			logger
			    ("[Error] there are problems with row or col in eval_jac_g.\n");
			PyErr_Print();
		}
		rowd = (long *) PyArray_DATA(row);
		cold = (long *) PyArray_DATA(col);

		for (i = 0; i < nele_jac; i++) {
			iRow[i] = (Index) rowd[i];
			jCol[i] = (Index) cold[i];
		}
		Py_CLEAR(arrayx);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_jac_g(1)");
	} else {
		PyObject *arglist;
		PyArrayObject *result;
		PyObject *arrayx =
		    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
					      (char *)x);

		if (!arrayx)
			return FALSE;

		if (new_x && myowndata->apply_new_python) {
			/* Call the python function to applynew */
			PyObject *arg1;
			PyObject *tempresult;

			arg1 = Py_BuildValue("(O)", arrayx);
			tempresult =
			    PyObject_CallObject(myowndata->apply_new_python,
						arg1);
			if (tempresult == NULL) {
				logger("[Error] Python function apply_new returns NULL");
				Py_DECREF(arg1);
				return FALSE;
			}
			Py_DECREF(arg1);
			Py_DECREF(tempresult);
		}

		if (user_data != NULL)
			arglist = Py_BuildValue("(OOO)",
						arrayx, Py_False,
						(PyObject *) user_data);
		else
			arglist = Py_BuildValue("(OO)", arrayx, Py_False);

		result = (PyArrayObject *) PyObject_CallObject(
        myowndata->eval_jac_g_python, arglist);

		if (result == NULL) {
      logger("[Error] Python function eval_jac_g returns NULL");
			PyErr_Print();
      return FALSE;
    }

    if (!PyArray_Check(result)) {
      logger("[Error] Python function eval_jac_g returns non-PyArray");
      Py_DECREF(result);
      return FALSE;
    }

		/*
		 * Code is buggy here. We assume that result is a double
		 * array
		 */
		assert(result->descr->type == 'd');
		tempdata = (double *) PyArray_DATA(result);

		for (i = 0; i < nele_jac; i++)
			values[i] = tempdata[i];

		Py_DECREF(result);
		Py_CLEAR(arrayx);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_jac_g(2)");
	}
	logger("[Callback:R] eval_jac_g");
	return TRUE;
}

Bool
eval_h(Index n, Number * x, Bool new_x, Number obj_factor,
       Index m, Number * lambda, Bool new_lambda,
       Index nele_hess, Index * iRow, Index * jCol,
       Number * values, UserDataPtr data)
{
	DispatchData *myowndata;
	UserDataPtr user_data;

	int i;
	npy_intp dims[1];
	npy_intp dims2[1];

	PyObject *result;
	size_t result_size;

	PyArrayObject *row;
	PyArrayObject *col;

	long *rdata;
	long *cdata;

	PyObject *lagrangex;

	PyObject *arglist;

	logger("[Callback:E] eval_h");

	myowndata = (DispatchData *) data;
	user_data = (UserDataPtr) myowndata->userdata;

	if (myowndata->eval_h_python == NULL) {
		logger("[Error] There is no eval_h assigned");
		return FALSE;
	}
	if (values == NULL) {
		PyObject *newx = Py_True;
		PyObject *lagrange = Py_True;
		PyObject *arglist;
		PyObject *objfactor;

		logger("[Callback:E] eval_h (1a)");

		objfactor = Py_BuildValue("d", obj_factor);

		if (user_data != NULL) {
			arglist = Py_BuildValue(
          "(OOOOO)", newx, lagrange, objfactor, Py_True,
          (PyObject *) user_data);
    } else {
			arglist = Py_BuildValue(
          "(OOOO)", newx, lagrange, objfactor, Py_True);
    }

    if (arglist == NULL) {
      logger("[Error] failed to build arglist for eval_h");
			PyErr_Print();
      return FALSE;
    } else {
      logger("[Logspam] built arglist for eval_h");
    }

	result = PyObject_CallObject(myowndata->eval_h_python, arglist);

    if (result == NULL) {
      logger("[Error] Python function eval_h returns NULL");
			PyErr_Print();
      return FALSE;
    } else {
      logger("[Logspam] Python function eval_h returns non-NULL");
    }

    result_size = PyTuple_Size(result);

    if (result_size == (size_t) -1) {
      logger("[Error] Python function eval_h returns non-PyTuple");
      Py_DECREF(result);
      return FALSE;
    }

    if (result_size != 2) {
      logger("[Error] Python function eval_h returns a tuple whose len != 2");
      Py_DECREF(result);
      return FALSE;
    }

    logger("[Callback:E] eval_h (tuple is the right length)");

	row = (PyArrayObject *) PyTuple_GetItem(result, 0);
	col = (PyArrayObject *) PyTuple_GetItem(result, 1);

	rdata = (long *) PyArray_DATA(row);
	cdata = (long *) PyArray_DATA(col);

	for (i = 0; i < nele_hess; i++) {
		iRow[i] = (Index) rdata[i];
		jCol[i] = (Index) cdata[i];
		/*
		 * logger("PyIPOPT_DEBUG %d, %d\n", iRow[i],
		 * jCol[i]);
		 */
	}

    logger("[Callback:E] eval_h (clearing stuff now)");

		Py_DECREF(objfactor);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_h (1b)");
	} else {
		PyObject *objfactor;
		PyObject *arrayx;
		PyArrayObject *result;
		double *tempdata;

		logger("[Callback:R] eval_h (2a)");

		objfactor = Py_BuildValue("d", obj_factor);

		dims[0] = n;
		arrayx =
		    PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
					      (char *)x);
		if (!arrayx)
			return FALSE;

		if (new_x && myowndata->apply_new_python) {
			/* Call the python function to applynew  */
			PyObject *arg1;
			PyObject *tempresult;
			arg1 = Py_BuildValue("(O)", arrayx);
			tempresult = PyObject_CallObject(
          myowndata->apply_new_python, arg1);
			if (tempresult == NULL) {
				logger("[Error] Python function apply_new returns NULL");
        PyErr_Print();
				Py_DECREF(arg1);
				return FALSE;
			}
			Py_DECREF(arg1);
			Py_DECREF(tempresult);
		}

		dims2[0] = m;
		lagrangex = PyArray_SimpleNewFromData(
        1, dims2, NPY_DOUBLE, (char *)lambda);
		if (!lagrangex)
			return FALSE;

		if (user_data != NULL) {
			arglist = Py_BuildValue(
          "(OOOOO)", arrayx, lagrangex, objfactor, Py_False,
          (PyObject *) user_data);
    	} else {
			arglist = Py_BuildValue(
			"(OOOO)", arrayx, lagrangex, objfactor, Py_False);
    	}
		result = (PyArrayObject *) PyObject_CallObject(
        myowndata->eval_h_python, arglist);

		if (result == NULL) {
			logger("[Error] Python function eval_h returns NULL");
			PyErr_Print();
			return FALSE;
		}

		if (!PyArray_Check(result)) {
			logger("[Error] Python function eval_h returns non-PyArray");
			Py_DECREF(result);
			return FALSE;
    	}

		tempdata = (double *) PyArray_DATA(result);

		for (i = 0; i < nele_hess; i++) {
			values[i] = tempdata[i];
			logger("PyDebug %lf", values[i]);
		}
		Py_CLEAR(arrayx);
		Py_CLEAR(lagrangex);
		Py_CLEAR(objfactor);
		Py_DECREF(result);
		Py_CLEAR(arglist);
		logger("[Callback:R] eval_h (2b)");
	}
	return TRUE;
}
