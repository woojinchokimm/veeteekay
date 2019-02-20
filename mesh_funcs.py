from vtk import *
import sys, os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from vtk.util import numpy_support
from mpl_toolkits.mplot3d.axes3d import Axes3D
from descartes import PolygonPatch
import pylab as pl
import shapely.geometry as geometry
import ipyvolume.pylab as p3
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
import glob
import os
import hickle as hkl
import networkx as nx
from sklearn.decomposition import PCA
from scipy.signal import argrelmin
from scipy.signal import find_peaks
import scipy.optimize
import scipy
from scipy import interpolate
import networkx as nx
from networkx.algorithms.components.connected import connected_components
from shapely import geometry, ops

class mesh():
    def __init__(self, opt):
        self.pers_var = None # personal option that will be used to facilitate debugging
        self.opt = opt  # type of data (CT or US)
        self.mesh_reader = None
        self.meshActor = None
        self.mesh_poly = None

        self.endoActor = None
        self.epiActor = None
        self.irActor = None
        self.orActor = None

        self.points = None
        self.triangles = None
        self.npts = None

        self.n_endo = None
        self.n_epi = None
        self.endoFilter = None
        self.epiFilter = None
        self.epi_poly = None
        self.endo_poly = None
        self.epi_apex_node = None
        self.endo_apex_node = None

        self.inner_rim_poly = None
        self.inner_rim_points = None
        self.inner_num_rim_pts = None
        self.outer_rim_poly = None
        self.outer_rim_points = None
        self.outer_num_rim_pts = None

        self.C = None  # center point of the rim
        self.P_fitcircle = None

        self.aorticValvePts = None
        self.mitralValvePts = None
        self.combinedPoints = None  # points that contains both the ring and the centerline
        self.aortic_line = None  # aortic points + line that divides aortic and mitral valve
        self.mitral_line = None
        self.aortic_cent = None
        self.mitral_cent = None
        self.com = None # center of mass of the entire mesh

        self.plane_pts = None  # the 3 points that define the 3rd chamber view
        self.original_planes = None
        self.axis_of_rot_normalized = None
        self.orig_view_angles = None #4 -> 2,3 rot angles (1D vector)

        self.orig_cut_poly_array = None #2D array: rows is 4ch--> 2ch--> 3ch and cols is endo, epi
        self.orig_planeActor = None #2D array: rows is 4ch--> 2ch--> 3ch and cols is endo, epi

        self.numSamples = None  # number of horizontal lines
        self.actual_num_samples = None
        self.L = None
        self.ideal_vol = None
        self.ground_truth_vol = None

        # the horizontal points computed from the ideal angles
        self.ideal_horiz_2ch_a = None
        self.ideal_horiz_2ch_b = None
        self.ideal_horiz_4ch_a = None
        self.ideal_horiz_4ch_b = None
        self.ideal_lowest_point_2ch = None
        self.ideal_lowest_point_4ch = None

        self.thickness = None
        self.rv_dir = None
        self.plane_colors = [(0, 255, 0), (0, 0, 255), (0.9100, 0.4100, 0.1700)] #4,2,3

        # save pca vectors
        self.pca1 = None
        self.pca2 = None
        self.pca3 = None

    #  import mesh and read files ..

    def input_mesh(self, vtkFile, display_opt):
        self.mesh_reader = vtk.vtkDataSetReader()
        self.mesh_reader.SetFileName(vtkFile)
        self.mesh_reader.ReadAllScalarsOn()  # Activate the reading of all scalars
        self.mesh_reader.Update()
        self.mesh_reader.GetHeader()
        self.mesh_poly = self.mesh_reader.GetOutput()

        meshMapper = vtk.vtkPolyDataMapper()
        meshMapper.SetInputConnection(self.mesh_reader.GetOutputPort())

        self.meshActor = vtk.vtkActor()
        self.meshActor.SetMapper(meshMapper)
        self.meshActor.GetProperty().SetColor(1.0, 0.0, 0.0)


        if (display_opt):
            axes = get_axes_actor([80,80,80], [0,0,0])

            renderer = vtk.vtkRenderer()
            renderer.SetBackground(1.0, 1.0, 1.0)
            renderer.AddActor(self.meshActor)
            renderer.AddActor(axes)
            vtk_show(renderer)

    def view_meshActor(self):
        axes = get_axes_actor([80,80,80], [0,0,0])

        ren = vtk.vtkRenderer()
        ren.AddActor(self.meshActor)
        ren.AddActor(axes)
        vtk_show(ren)

    def set_points_verts(self):

        self.triangles = self.mesh_poly.GetPolys().GetData()
        self.points = self.mesh_poly.GetPoints()

    def align_meshes_pca(self, display_opt):
        """
            Aligns mesh into the main pcs.
            !! This makes x-axis the first pc !!
        """
        # convert vtk points to numpy first
        vtk_pts = self.points
        numpy_pts = numpy_support.vtk_to_numpy(vtk_pts.GetData())

        # perform pca
        pca = PCA(n_components=3)
        trans_coords = pca.fit_transform(numpy_pts)
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_ratio_

        # save pca vectors as global variables
        self.pca1 = eigenvectors[0]
        self.pca2 = eigenvectors[1]
        self.pca3 = eigenvectors[2]

        if display_opt:
            axes = get_axes_actor([80,80,80], [0,0,0])

            trans_act = include_points(trans_coords, trans_coords.shape[0], 4, (0,1,0))
            self.meshActor.GetProperty().SetOpacity(0.6)

            ren = vtk.vtkRenderer()
            ren.AddActor(self.meshActor)
            ren.AddActor(trans_act)
            ren.AddActor(axes)
            vtk_show(ren)

        # reset the self.attributes with transformed coordinates
        trans_vtk_pts = MakevtkPoints(trans_coords, deep=True)
        self.points = trans_vtk_pts
        self.mesh_poly.SetPoints(trans_vtk_pts)

        meshMapper = vtk.vtkPolyDataMapper()
        meshMapper.SetInputData(self.mesh_poly)

        self.meshActor = vtk.vtkActor()
        self.meshActor.SetMapper(meshMapper)
        self.meshActor.GetProperty().SetColor(1.0, 0.0, 0.0)

    def add_endo_epi_labels(self):
        if (self.opt != 'US'):
            print('Please use "get_endo_epi_rim_data()" for CT data.')

        filename = 'D:\\Documents\\UNI\\MRes in Medical Imaging\\PhD Project\\Esther functions\\epiendo_icl.txt'
        endo_epi_arr = np.loadtxt(filename)
        self.npts = self.mesh_poly.GetNumberOfPoints()

        # check that the epi_endo_labels correspond to the icl vtk file format (number of points should match)
        if len(endo_epi_arr) != self.npts:
            print('epiendo data does not correspond to the current vtk file!')

        # label each point with endo or epi
        endo_epi_labels = numpy_support.numpy_to_vtk(
            num_array=endo_epi_arr, deep=True, array_type=vtk.VTK_DOUBLE)
        endo_epi_labels.SetName("endoepilabels")
        self.mesh_poly.GetPointData().AddArray(endo_epi_labels)

        # how to extract specific label values with the indexing of the original pts array
        arr = vtk.vtkDoubleArray()
        arr = self.mesh_poly.GetPointData().GetAbstractArray("endoepilabels")

        self.n_endo = int(np.sum(endo_epi_arr))  # number of endo points
        self.n_epi = int(self.npts - self.n_endo)  # number of epi point

    def separate_endo_epi(self, threshold, display_opt):
        epi_thresh = vtk.vtkThreshold()  # for smooth surface that includes triangulation data
        # epi_thresh = vtk.vtkThresholdPoints()
        epi_thresh.ThresholdByLower(threshold)  # epi = 0
        epi_thresh.SetInputArrayToProcess(0, 0, 0, 0, "endoepilabels")
        epi_thresh.SetInputData(self.mesh_poly)
        epi_thresh.Update()

        # create geometry filter
        self.epiFilter = vtk.vtkGeometryFilter()
        self.epiFilter.SetInputData(epi_thresh.GetOutput())
        self.epiFilter.Update()

        # epi_poly
        self.epi_poly = self.epiFilter.GetOutput()

        # create poly data mapper
        epiMapper = vtk.vtkPolyDataMapper()
        epiMapper.SetInputConnection(self.epiFilter.GetOutputPort())

        # create actor
        epiActor = vtk.vtkActor()
        epiActor.SetMapper(epiMapper)
        epiActor.GetProperty().SetColor(1.0, 0.0, 0.0)
        epiActor.GetProperty().SetLineWidth(2)

        #######################################################################################################################

        # create thresholder for ENDO
        endo_thresh = vtk.vtkThreshold()  # for smooth surface that includes triangulation data
        # endo_thresh = vtk.vtkThresholdPoints() # for points rather than smooth surface
        endo_thresh.ThresholdByUpper(threshold)  # endo = 1
        endo_thresh.SetInputArrayToProcess(0, 0, 0, 0, "endoepilabels")
        endo_thresh.SetInputData(self.mesh_poly)
        endo_thresh.Update()

        # create geometry filter
        self.endoFilter = vtk.vtkGeometryFilter()
        self.endoFilter.SetInputData(endo_thresh.GetOutput())
        self.endoFilter.Update()

        self.endo_poly = self.endoFilter.GetOutput()

        # create poly data mapper
        endoMapper = vtk.vtkPolyDataMapper()
        endoMapper.SetInputConnection(self.endoFilter.GetOutputPort())

        # create actor
        endoActor = vtk.vtkActor()
        endoActor.SetMapper(endoMapper)
        endoActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        endoActor.GetProperty().SetLineWidth(2)

        ########################################################################################################################

        # generate renderer
        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.AddActor(epiActor)
        ren.AddActor(endoActor)

        # display
        if (display_opt):
            vtk_show(ren)

    def set_apex_node(self):
        """
        Sets epi and endo apex nodes based on prior knowledge.
        """
        if self.opt == 'CT':
            self.epi_apex_node = self.mesh_poly.GetPoints().GetPoint(3604)
            self.endo_apex_node = self.mesh_poly.GetPoints().GetPoint(3579)
        else:
            self.endo_apex_node = None  # we do not know this
            self.epi_apex_node = self.mesh_poly.GetPoints().GetPoint(0)

    def compute_apex_nodes(self, display_opt):
        """
            Function to compute apex nodes (for both endo and epi).
            For these, we need to align the pca.
            Finds the points with the lowest x-component (corresponding to pca1)
        """
        # convert endo and epi poly to numpys for faster processing:
        numpy_endo = numpy_support.vtk_to_numpy(self.endo_poly.GetPoints().GetData())
        numpy_epi = numpy_support.vtk_to_numpy(self.epi_poly.GetPoints().GetData())

        # simply find lowest pca1 component since they all aligned
        min_endo_idx = np.argmin(numpy_endo[:,0])
        min_epi_idx = np.argmin(numpy_epi[:,0])

        endo_minima = numpy_endo[min_endo_idx]
        epi_minima = numpy_epi[min_epi_idx]

        if display_opt:
            minima_endo_act = include_points(list(endo_minima), 1, 10, (1,1,1))
            minima_epi_act = include_points(list(epi_minima), 1, 10, (1,1,1))

            self.endoActor.GetProperty().SetOpacity(0.6)
            self.epiActor.GetProperty().SetOpacity(1)

            axes = get_axes_actor([200,50,50], [-0.51,0,0])
            ren = vtk.vtkRenderer()
            ren.SetBackground(0,0,0)
            ren.AddActor(self.endoActor)
            ren.AddActor(self.epiActor)
            ren.AddActor(minima_endo_act)
            ren.AddActor(minima_epi_act)
            ren.AddActor(axes)
            vtk_show(ren)

        self.endo_apex_node = endo_minima
        self.epi_apex_node = epi_minima

    def get_endo_epi_rim_data(self, fnms):
        if (self.opt != 'CT'):
            print('Please use "add_endo_epi_labels()" for US mesh.')

        endo_fnm = fnms[0]
        epi_fnm = fnms[1]
        inner_rim_fnm = fnms[2]
        outer_rim_fnm = fnms[3]

        # import the text file and set all numbers as int (since we only have ids)
        endo_arr = np.genfromtxt(endo_fnm, delimiter=",", skip_header=1, dtype=int)
        epi_arr = np.genfromtxt(epi_fnm, delimiter=",", skip_header=1, dtype=int)
        inner_rim_arr = np.genfromtxt(inner_rim_fnm, delimiter=",", skip_header=1, dtype=int)
        outer_rim_arr = np.genfromtxt(outer_rim_fnm, delimiter=",", skip_header=1, dtype=int)

        # delete the first column of endo and epi arrays
        endo_arr = endo_arr[:, 4:]
        epi_arr = epi_arr[:, 4:]
        inner_rim_arr = inner_rim_arr[:, 3:]
        self.inner_num_rim_pts = inner_rim_arr.shape[0]
        outer_rim_arr = outer_rim_arr[:, 3:]

        # create the polydata from these arrays
        self.endo_poly, self.inner_rim_poly = self.create_polydata_from_id_array(endo_arr, inner_rim_arr)
        self.epi_poly, self.outer_rim_poly = self.create_polydata_from_id_array(epi_arr, outer_rim_arr)


    def endo_epi_rim_model(self, fnms):
        if (self.opt != 'model'):
            print('Please check what type of mesh you have. This function only works for model meshes.')

        epi_fnm = fnms[0]
        endo_fnm = fnms[1]
        inner_rim_fnm = fnms[2]

        # import the text file and set all numbers as int (since we only have ids)
        epi_arr = np.genfromtxt(epi_fnm, delimiter=",", skip_header=1, dtype=int)
        endo_arr = np.genfromtxt(endo_fnm, delimiter=",", skip_header=1, dtype=int)
        inner_rim_arr = np.genfromtxt(inner_rim_fnm, delimiter=",", skip_header=1, dtype=int)

        # delete all but first and last columns
        epi_arr = epi_arr[:, [9, 10]]
        endo_arr = endo_arr[:, [9, 10]]
        inner_rim_arr = inner_rim_arr[:, [7, 8]]
        self.inner_num_rim_pts = inner_rim_arr.shape[0]

        # create the polydata from these arrays
        self.epi_poly = self.create_polydata_from_id_array(epi_arr, 0)
        self.endo_poly = self.create_polydata_from_id_array(endo_arr, 0)
        self.inner_rim_poly = self.create_polydata_from_id_array(inner_rim_arr, 1)

    def create_polydata_from_id_array(self, arr, rim_arr):
        """  This functions creates polydata objects

        Inputs:
            arr : the information on cell ids and point ids
            typ : 0 is polygons are required (for endo and epi poly)
                  1 means only points are required (for rim)
            rim_arr : contains pointid information (rather than cellid)

        Output:
            polydata (labelled for endo/epi_poly)
        """

        # for endo/epi_poly (using cellid data in arr)
        tot_num_cells = arr.shape[0]
        selected_idx = []
        for i in range(tot_num_cells):
            if arr[i, 1] == 1:  # if vtkIsSelected == True
                selected_idx.append(arr[i, 0])

        num_selected = len(selected_idx)

        polydata = vtk.vtkPolyData()

        points = self.points  # you need to include all the points from the original data set, the only thing that differs is which triangles are shown
        triangles = vtk.vtkCellArray()
        for i in range(num_selected):
            cellId = selected_idx[i]
            cell = self.mesh_poly.GetCell(cellId)
            triangles.InsertNextCell(cell)

        # create the polydata
        # you need to put it ALL the points of the mesh (not just the ones labelled)
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        # make sure now you clean the polydata to remove unused points
        cpd = vtk.vtkCleanPolyData()
        cpd.SetInputData(polydata)
        cpd.Update()

        polydata = cpd.GetOutput() #endo_poly or epi_poly

        # now label endo or epi_poly with rim_poly points
        labels = vtk.vtkIntArray()
        labels.SetNumberOfComponents(1)
        labels.SetName("rim labels")

        # for rim points
        tot_num_cells = rim_arr.shape[0]
        selected_idx = []
        for i in range(tot_num_cells):
            if rim_arr[i, 1] == 1:  # if vtkIsSelected == True
                selected_idx.append(rim_arr[i, 0])

        num_selected = len(selected_idx)

        rimpoly = vtk.vtkPolyData() # rim polydata
        points = vtk.vtkPoints()
        for i in range(num_selected):
            pointId = selected_idx[i]
            pt = self.mesh_poly.GetPoints().GetPoint(pointId)
            points.InsertNextPoint(pt)
            labels.InsertNextValue(23)

        # insert label as field data
        polydata.GetFieldData().AddArray(labels)

        rimpoly.SetPoints(points)

        return polydata, rimpoly

    def actor_from_segment(self, polydata, typ, display_opt):
        """
        Converts polydata segment (e.g. epi_poly, endo_poly, inner_rim_poly, ...etc.)
        into actor and displays it.
        """
        # if opt = 0: polydata has only point data (i.e. requires vertex filter)
        # if opt = 1: polydata has point + cell data

        self.meshActor.GetProperty().SetOpacity(0.3)

        if typ == 0:  # point data
            # add vertex at each point (so no need to insert vertices manually)
            vertexFilter = vtk.vtkVertexGlyphFilter()
            vertexFilter.SetInputData(polydata)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertexFilter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)
            actor.GetProperty().SetPointSize(7)

            if display_opt:
                axes = get_axes_actor([80,80,80], [0,0,0])
                ren = vtk.vtkRenderer()
                ren.SetBackground(1.0, 1.0, 1.0)
                ren.AddActor(self.meshActor)
                ren.AddActor(axes)
                ren.AddActor(actor)
                vtk_show(ren)

        else:  # point data + cell data

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0, 1, 0)
            actor.GetProperty().SetPointSize(7)

            if display_opt:
                axes = get_axes_actor([80,80,80], [0,0,0])
                ren = vtk.vtkRenderer()
                ren.SetBackground(1.0, 1.0, 1.0)
                ren.AddActor(actor)
                ren.AddActor(self.meshActor)
                ren.AddActor(axes)
                vtk_show(ren)

        return actor

    #  extract the rim, aortic and mitral valve centers ..

    def extract_rim(self, display_opt):

        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(self.epi_poly)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.Update()

        self.inner_rim_poly = fe.GetOutput()
        self.inner_rim_points = self.inner_rim_poly.GetPoints()
        self.inner_num_rim_pts = self.inner_rim_points.GetNumberOfPoints()

        eM = vtk.vtkPolyDataMapper()
        eM.SetInputConnection(fe.GetOutputPort())

        eA = vtk.vtkActor()
        eA.GetProperty().SetColor(0.0, 0.0, 1.0)
        eA.GetProperty().SetLineWidth(5)
        eA.SetMapper(eM)

        # create poly data mapper to display the epi layer
        epiMapper = vtk.vtkPolyDataMapper()
        epiMapper.SetInputData(self.epi_poly)

        # create actor
        epiActor = vtk.vtkActor()
        epiActor.SetMapper(epiMapper)
        epiActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        epiActor.GetProperty().SetLineWidth(2)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.AddActor(epiActor)
        ren.AddActor(eA)

        if (display_opt):
            vtk_show(ren)

    def circle_fit(self, display):
        # -------------------------------------------------------------------------------
        # (1) Fitting plane by SVD for the mean-centered data
        # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
        # -------------------------------------------------------------------------------

        x_rim, y_rim, z_rim = get_xyz_vecs(self.inner_rim_points, self.inner_num_rim_pts)

        P = np.vstack((x_rim, y_rim, z_rim))
        P = np.transpose(P)

        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U, s, V = np.linalg.svd(P_centered)

        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2, :]
        d = -np.dot(P_mean, normal)  # d = -<p,n>

        # -------------------------------------------------------------------------------
        # (2) Project points to coords X-Y in 2D plane
        # -------------------------------------------------------------------------------
        P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

        # -------------------------------------------------------------------------------
        # (3) Fit circle in new 2D coords
        # -------------------------------------------------------------------------------
        xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

        # --- Generate circle points in 2D
        t = np.linspace(0, 2*np.pi, 100)
        xx = xc + r*np.cos(t)
        yy = yc + r*np.sin(t)

        # -------------------------------------------------------------------------------
        # (4) Transform circle center back to 3D coords
        # -------------------------------------------------------------------------------
        C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
        C = C.flatten()

        # --- Generate points for fitting circle
        t = np.linspace(0, 2*np.pi, 100)
        u = P[0] - C
        P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

        # --- Generate points for fitting arc
        u = P[0] - C
        v = P[-1] - C
        theta = angle_between(u, v, normal)

        if display:
            fig = plt.figure(figsize=(15, 11))
            alpha_pts = 0.5
            figshape = (2, 3)
            ax = [None]*4
            ax[0] = plt.subplot2grid(figshape, loc=(0, 0), colspan=2)
            ax[1] = plt.subplot2grid(figshape, loc=(1, 0))
            ax[2] = plt.subplot2grid(figshape, loc=(1, 1))
            ax[3] = plt.subplot2grid(figshape, loc=(1, 2))
            i = 0
            ax[i].set_title('Fitting circle in 2D coords projected onto fitting plane')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_aspect('equal', 'datalim')
            ax[i].margins(.1, .1)
            ax[i].grid()
            i = 1
            ax[i].scatter(P[:, 0], P[:, 1], alpha=alpha_pts, label='Cluster points P')
            ax[i].set_title('View X-Y')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_aspect('equal', 'datalim')
            ax[i].margins(.1, .1)
            ax[i].grid()
            i = 2
            ax[i].scatter(P[:, 0], P[:, 2], alpha=alpha_pts, label='Cluster points P')
            ax[i].set_title('View X-Z')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('z')
            ax[i].set_aspect('equal', 'datalim')
            ax[i].margins(.1, .1)
            ax[i].grid()
            i = 3
            ax[i].scatter(P[:, 1], P[:, 2], alpha=alpha_pts, label='Cluster points P')
            ax[i].set_title('View Y-Z')
            ax[i].set_xlabel('y')
            ax[i].set_ylabel('z')
            ax[i].set_aspect('equal', 'datalim')
            ax[i].margins(.1, .1)
            ax[i].grid()

            ax[0].scatter(P_xy[:, 0], P_xy[:, 1], alpha=alpha_pts, label='Projected points')

            ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle')
            ax[0].plot(xc, yc, 'k+', ms=10)
            ax[0].legend()

            ax[1].plot(P_fitcircle[:, 0], P_fitcircle[:, 1], 'k--', lw=2, label='Fitting circle')
            ax[2].plot(P_fitcircle[:, 0], P_fitcircle[:, 2], 'k--', lw=2, label='Fitting circle')
            ax[3].plot(P_fitcircle[:, 1], P_fitcircle[:, 2], 'k--', lw=2, label='Fitting circle')
            ax[3].legend()

            ax[1].plot(C[0], C[1], 'k+', ms=10)
            ax[2].plot(C[0], C[2], 'k+', ms=10)
            ax[3].plot(C[1], C[2], 'k+', ms=10)
            ax[3].legend()

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot(*P.T, ls='', marker='o', alpha=0.5, label='Cluster points P')

            # --- Plot fitting plane
            xx, yy = np.meshgrid(np.linspace(-40, 20, 100), np.linspace(-40, 40, 100))
            zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
            ax.plot_surface(xx, yy, zz, rstride=2, cstride=2, color='y', alpha=0.2, shade=False)

            # --- Plot fitting circle
            ax.plot(*P_fitcircle.T, color='k', ls='--', lw=2, label='Fitting circle')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            ax.set_aspect('equal', 'datalim')
            set_axes_equal_3d(ax)

        self.C = C
        self.P_fitcircle = P_fitcircle

    def set_combinedPoints(self, display_opt):

        spec1 = [self.inner_rim_points.GetPoint(69)[0], self.inner_rim_points.GetPoint(69)[
            1], self.inner_rim_points.GetPoint(69)[2]]
        spec2 = [self.inner_rim_points.GetPoint(110)[0], self.inner_rim_points.GetPoint(110)[
            1], self.inner_rim_points.GetPoint(110)[2]]

        # generate random points between the points (rather than creating a line)
        samp = np.random.uniform(low=0, high=1, size=(100,))

        x_samp = spec1[0] + samp*(spec2[0] - spec1[0])
        y_samp = spec1[1] + samp*(spec2[1] - spec1[1])
        z_samp = spec1[2] + samp*(spec2[2] - spec1[2])

        self.combinedPoints = vtk.vtkPoints()

        for i in range(self.inner_num_rim_pts):  # third add the points from the rim points
            self.combinedPoints.InsertNextPoint(self.inner_rim_points.GetPoint(i))

        # create polydata for the combined points
        combinedPoly = vtk.vtkPolyData()
        combinedPoly.SetPoints(self.combinedPoints)

        # add vertex at each point (so no need to insert vertices manually)
        vertexFilter = vtk.vtkVertexGlyphFilter()
        vertexFilter.SetInputData(combinedPoly)

        # create pipeline for display
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputConnection(vertexFilter.GetOutputPort())

        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetLineWidth(5)
        lineActor.GetProperty().SetPointSize(5)
        lineActor.GetProperty().SetColor(0, 0, 0)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.AddActor(lineActor)

        if (display_opt):
            vtk_show(ren)

    def sort_mitral_aortic_valve_points(self):

        self.aorticValvePts = vtk.vtkPoints()
        aortic_idx = np.arange(42)
        aortic_idx = np.append(aortic_idx, np.arange(70, 78))
        aortic_idx = np.append(aortic_idx, 112)

        for i in range(aortic_idx.shape[0]):
            self.aorticValvePts.InsertNextPoint(self.combinedPoints.GetPoint(aortic_idx[i]))

        self.mitralValvePts = vtk.vtkPoints()
        mitral_idx = np.arange(42, 70)
        mitral_idx = np.append(mitral_idx, np.arange(78, 110))
        mitral_idx = np.append(mitral_idx, [110, 111, 113, 114, 115, 116, 117, 118])

        for i in range(mitral_idx.shape[0]):
            self.mitralValvePts.InsertNextPoint(self.combinedPoints.GetPoint(mitral_idx[i]))

    def set_line_pts(self):
        self.aortic_line = vtk.vtkPoints() # DO FOR AORTIC

        spec1 = [self.inner_rim_points.GetPoint(69)[0], self.inner_rim_points.GetPoint(69)[
            1], self.inner_rim_points.GetPoint(69)[2]]
        spec2 = [self.inner_rim_points.GetPoint(110)[0], self.inner_rim_points.GetPoint(110)[
            1], self.inner_rim_points.GetPoint(110)[2]]

        # generate random points between the points (rather than creating a line)
        samp = np.random.uniform(low=0, high=1, size=(50,))

        x_samp = spec1[0] + samp*(spec2[0] - spec1[0])
        y_samp = spec1[1] + samp*(spec2[1] - spec1[1])
        z_samp = spec1[2] + samp*(spec2[2] - spec1[2])

        self.aortic_line.InsertNextPoint(spec1)  # first add the special point 1 (69)
        self.aortic_line.InsertNextPoint(spec2)  # first add the special point 2 (110)

        for i in range(x_samp.shape[0]):  # second add the points from the random sampling of the line
            self.aortic_line.InsertNextPoint([x_samp[i], y_samp[i], z_samp[i]])

        for i in range(self.aorticValvePts.GetNumberOfPoints()):  # third add the points from the rim points
            self.aortic_line.InsertNextPoint(self.aorticValvePts.GetPoint(i))

        # DO FOR MITRAL
        self.mitral_line = vtk.vtkPoints()

        spec1 = [self.inner_rim_points.GetPoint(69)[0], self.inner_rim_points.GetPoint(69)[
            1], self.inner_rim_points.GetPoint(69)[2]]
        spec2 = [self.inner_rim_points.GetPoint(110)[0], self.inner_rim_points.GetPoint(110)[
            1], self.inner_rim_points.GetPoint(110)[2]]

        # generate random points between the points (rather than creating a line)
        samp = np.random.uniform(low=0, high=1, size=(50,))

        x_samp = spec1[0] + samp*(spec2[0] - spec1[0])
        y_samp = spec1[1] + samp*(spec2[1] - spec1[1])
        z_samp = spec1[2] + samp*(spec2[2] - spec1[2])

        self.mitral_line.InsertNextPoint(spec1)  # first add the special point 1 (69)
        self.mitral_line.InsertNextPoint(spec2)  # first add the special point 2 (110)

        for i in range(x_samp.shape[0]):  # second add the points from the random sampling of the line
            self.mitral_line.InsertNextPoint([x_samp[i], y_samp[i], z_samp[i]])

        for i in range(self.mitralValvePts.GetNumberOfPoints()):  # third add the points from the rim points
            self.mitral_line.InsertNextPoint(self.mitralValvePts.GetPoint(i))

    def convex_hull_center(self, display_opt):

        mtN = int(self.aortic_line.GetNumberOfPoints()) # do for aortic line first
        mt_x = np.zeros((mtN), dtype=float)
        mt_y = np.zeros((mtN), dtype=float)
        mt_z = np.zeros((mtN), dtype=float)

        for i in range(mtN):
            mt_x[i] = self.aortic_line.GetPoint(i)[0]
            mt_y[i] = self.aortic_line.GetPoint(i)[1]
            mt_z[i] = self.aortic_line.GetPoint(i)[2]

        mtX = np.column_stack((mt_x, mt_y, mt_z))

        point_collection = geometry.MultiPoint(list(np.column_stack((mtX[:, 0], mtX[:, 1]))))
        point_collection.envelope

        convex_hull_polygon = point_collection.convex_hull
        cent = np.array(convex_hull_polygon.centroid)

        if display_opt:
            _ = plot_polygon(convex_hull_polygon)
            _ = pl.plot(mtX[:, 0], mtX[:, 1], 'o', color='#f16824')
            _ = pl.plot(cent[0], cent[1], 'o', color='red')

        # for z-component of polygon center, just take the average (avcent[2]) (a bit inaccurate but acceptable since rim is almost flat)
        # tkae x and x components from (cent) as it is more accurate

        self.aortic_cent = np.array([cent[0], cent[1], np.mean(mtX[:, 2])])

        # do for mitral line first
        mtN = int(self.mitral_line.GetNumberOfPoints())
        mt_x = np.zeros((mtN), dtype=float)
        mt_y = np.zeros((mtN), dtype=float)
        mt_z = np.zeros((mtN), dtype=float)

        for i in range(mtN):
            mt_x[i] = self.mitral_line.GetPoint(i)[0]
            mt_y[i] = self.mitral_line.GetPoint(i)[1]
            mt_z[i] = self.mitral_line.GetPoint(i)[2]

        mtX = np.column_stack((mt_x, mt_y, mt_z))

        point_collection = geometry.MultiPoint(list(np.column_stack((mtX[:, 0], mtX[:, 1]))))
        point_collection.envelope

        convex_hull_polygon = point_collection.convex_hull
        cent = np.array(convex_hull_polygon.centroid)

        if display_opt:
            _ = plot_polygon(convex_hull_polygon)
            _ = pl.plot(mtX[:, 0], mtX[:, 1], 'o', color='#f16824')
            _ = pl.plot(cent[0], cent[1], 'o', color='red')

        self.mitral_cent = np.array([cent[0], cent[1], np.mean(mtX[:, 2])])

    def set_rv_dir(self, rv_dirs, iter):
        if np.isscalar(rv_dirs):
            print('no rv_dir given, using +y axis as RV dir ..')
            self.rv_dir = np.array([0,1,0])
        else:
            self.rv_dir = rv_dirs[iter]
            print('rv given : ', self.rv_dir)

    #   extract 2,3 and 4 apical chamber views..

    def mesh_slicer(self, plane, opt):
        """Function to obtain slice of 3D mesh from given plane (using intersection of point to plane)

        Args:
            mesh_reader (vtkObject) : the mesh read from using vtk pipeline
            plane (tuple) : coefficients of the plane (a,b,c,d)
            points (list of tuples) : nodes of the mesh
            opt (string) : {'cut','all'}  -->  display only the cut slice or the mesh with outline of cut through plane

        Returns:
            rot_plane : the rotated plane
        """

        # get plane coefficients
        a = plane[0]
        b = plane[1]
        c = plane[2]

        # create vtk plane object
        VTKplane = vtk.vtkPlane()
        # for now we choose the center point as the point of rotation
        VTKplane.SetOrigin(self.mesh_poly.GetCenter())
        VTKplane.SetNormal(a, b, c)
        VTKplane.SetOrigin(self.epi_apex_node)

        # create cutter
        cutEdges = vtk.vtkCutter()
        cutEdges.SetInputData(self.mesh_poly)
        cutEdges.SetCutFunction(VTKplane)
        cutEdges.GenerateCutScalarsOn()
        cutEdges.SetValue(0, 0.5)

        # create renderer
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.0, 0.0, 0.0)

        # create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutEdges.GetOutputPort())

        # create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetLineWidth(2)

        # display apex point
        apexA = include_points(list(self.epi_apex_node), 1, 15, (0, 0, 1))

        if (opt == 'mesh'):
            meshMapper = vtk.vtkPolyDataMapper()
            meshMapper.SetInputData(self.mesh_poly)
            meshActor = vtk.vtkActor()
            meshActor.SetMapper(meshMapper)
            meshActor.GetProperty().SetColor(1.0, 0.0, 0.0)

            # generate renderer
            ren.AddActor(self.meshActor)
            ren.AddActor(actor)
            ren.AddActor(apexA)

        else:
            ren.AddActor(actor)
            ren.AddActor(apexA)

        # display
        vtk_show(ren)

    def set_orig_view_angles(self, orig_view_angles):
        self.orig_view_angles = orig_view_angles

    def set_center_irp(self):
        # step 1: find center of inner rim poly
        numpy_irp = numpy_support.vtk_to_numpy(self.inner_rim_poly.GetPoints().GetData())
        self.C  = np.mean(numpy_irp, axis=0)

    def find_4ch_view(self, display_opt):
        """ +y dir points towards RV -> use this as 4ch view
            more reliable than finding mitral and aortic valve points

            the three points on the 4ch view are:
                1) center of inner rim poly
                2) apex point
                3) the y direction
        """
        # step 1: check if self.C exists
        if self.C is None:
            print('Center of inner rim poly is not set! exiting..')
            sys.exit()

        # step 2: find y direction
        pt_rv_dir = 50.0*self.rv_dir

        # set plane_pts :
        self.plane_pts = np.vstack((self.C, pt_rv_dir, self.epi_apex_node))

        # construct plane using p1, p2 and the apex node
        four_ch_view_plane_normal = find_plane_eq(self.C, pt_rv_dir, self.epi_apex_node)

        if display_opt:
            # display x-y-z actor
            axes = get_axes_actor([50,50,50], [0,0,0])

            c_irp_act = include_points(self.C, 1, 10, (1,0,1))
            pt_rv_dir_act = include_points(pt_rv_dir, 1, 10, (1,1,0))
            epi_apex_act = include_points(list(self.epi_apex_node), 1, 10, (1,0,1))

            ren = vtk.vtkRenderer()
            ren.AddActor(self.meshActor)
            ren.AddActor(c_irp_act)
            ren.AddActor(epi_apex_act)
            ren.AddActor(pt_rv_dir_act)
            ren.AddActor(axes)
            vtk_show(ren)
        # # step 1: find center of endo_poly
        # endo_numpy = numpy_support.vtk_to_numpy(self.endo_poly.GetPoints().GetData())
        # com = np.mean(endo_numpy, 0)
        #
        # # step 2: construct line rv_dir that is translated at position com
        # pSource = com - 100*self.rv_dir
        # pTarget = com + 100*self.rv_dir

        # # step 3: find intersection of line with endo_poly
        # bspTree = vtk.vtkModifiedBSPTree()
        # bspTree.SetDataSet(self.endo_poly)  # cut through endo polydata (not mesh)
        # bspTree.BuildLocator()
        #
        # # set these as plane_pts
        # p1 = pSource
        # p2 = pTarget
        #
        # four_ch_valve_pts = [p1, p2]
        # self.plane_pts = np.vstack((p1, p2, self.epi_apex_node))
        #
        # # construct plane using p1, p2 and the apex node
        # four_ch_view_plane_normal = find_plane_eq(p1, p2, self.epi_apex_node)

        # if display_opt:
        #     # display x-y-z actor
        #     axes = get_axes_actor([80,80,80], [0,0,0])
        #
        #     p1_act = include_points(p1, 1, 10, (1,0,1))
        #     p2_act = include_points(p2, 1, 10, (1,1,0))
        #     epi_apex_act = include_points(list(self.epi_apex_node), 1, 10, (1,0,1))
        #     endo_apex_act = include_points(list(self.endo_apex_node), 1, 10, (1,0,1))
        #
        #     ren = vtk.vtkRenderer()
        #     ren.AddActor(self.meshActor)
        #     ren.AddActor(p1_act)
        #     ren.AddActor(p2_act)
        #     ren.AddActor(axes)
        #     ren.AddActor(epi_apex_act)
        #     ren.AddActor(endo_apex_act)
        #
        #     vtk_show(ren)

            # display the 4-ch view
            _ = self.mesh_slicer(four_ch_view_plane_normal, 'mesh')

        return four_ch_view_plane_normal

    def set_original_planes(self, display_opt):
        """
        Function computes the planes for 4, 2 and 3 chamber views.

        How do we set the LV midline (i.e. the axis of rotation)?

        In clinical practice: "straight line was traced between the attachment points of the mitral annulus with the valve leaflets."
             see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929292/

        Since we do not have information on ventricles, Paul says
        to simply take midline as apex to center of inner rim points.
        """

        # get 4-chamber view
        four_ch_view_plane_normal = self.find_4ch_view(display_opt)

        # set rodriguez rotation around midline (apex to C)
        axis_of_rot = np.array(self.epi_apex_node - self.C)
        self.axis_of_rot_normalized = axis_of_rot/np.linalg.norm(axis_of_rot)

        # get 2-chamber view (90-counterclock rotation from 4ch)
        new_P = my_rodriguez_rotation(self.plane_pts, self.axis_of_rot_normalized,
            math.radians(self.orig_view_angles[1]))  # rodriguez rotation around midline
        two_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

        # get 3-chamber view (additional 30-60 counterclock rotation from 3ch)
        new_P = my_rodriguez_rotation(self.plane_pts, self.axis_of_rot_normalized, math.radians(self.orig_view_angles[2]))
        three_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

        if display_opt:
            _ = self.mesh_slicer(four_ch_view_plane_normal, 'mesh')
            _ = self.mesh_slicer(two_ch_view_plane_normal, 'mesh')
            _ = self.mesh_slicer(three_ch_view_plane_normal, 'mesh')

        self.original_planes = np.vstack((four_ch_view_plane_normal,
                                            two_ch_view_plane_normal,
                                            three_ch_view_plane_normal))

    def get_cut_poly_array(self, planes, angles, disp, fix_pts):
        """Function to obtain slice of 3D mesh from given plane (using intersection of point to plane)
            Cuts through endocardium only (not the entire mesh)
        Args:
            planes (2D array) : coefficients of the planes (a | b | c | d) --> each row is a new plane
                              : Is in order 4ch --> 2ch --> 3ch
            offset (float array): 4ch and 2ch specific angles (variability test) or specific offsets (foreshortening test)
            disp (boolean) : 0 for no display, 1 for display
            fix_pts (boolean,point OR boolean, 3 points) : for variability test, we need to make sure that the views are fixed to the apex_node, but for foreshortening test we fix on to the varying offsets!
        Returns:
            cut_poly_array : array 3x2 where rows is 4,2,3ch and columns is endo, epi.
            Displays of the chamber views in different colours
        """
        noPlanes = len(planes)
        plane_storer = [] #4, 2, 3ch in this order
        cut_poly_array = []  #4, 2, 3ch in this order

        view_type = ['4ch', '2ch', '3ch']

        for i in range(noPlanes):
            if fix_pts[0] == 'var':  # for variability test
                origin = self.epi_apex_node
            else:  # for foreshortening test
                origin = fix_pts[1+i]

            cutPoly_endo_epi, planeActor_endo_epi = self.get_edges_strips(planes[i], origin,
                                                                        view_type[i], self.plane_colors[i])
            cut_poly_array.append(cutPoly_endo_epi) # 4, 2, 3
            plane_storer.append(planeActor_endo_epi)


        #           DISPLAY PURPOSES        #

        # include apex_node
        apexA = include_points(list(self.epi_apex_node), 1, 15, (0, 0, 0))

        ## create legend box ##
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(3)

        legendBox = vtk.vtkCubeSource()
        legendBox.SetXLength(2)
        legendBox.SetYLength(2)
        legend.SetEntry(0, legendBox.GetOutput(), "4 ch", (0, 1, 0)) #green
        legend.SetEntry(1, legendBox.GetOutput(), "2 ch", (0, 0, 1)) #blue

        legend.UseBackgroundOn()
        legend.LockBorderOn()
        legend.SetBackgroundColor(0.5, 0.5, 0.5)

        # create text box to display the angles ..
        textActor = vtk.vtkTextActor()
        textActor.SetInput("4ch = " + str(angles[0])
                    + "\n" + "2ch = " + str(angles[1]))
        textActor.SetPosition2(10, 40)
        textActor.GetTextProperty().SetFontSize(24)
        textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)

        # display x-y-z actor
        axes = get_axes_actor([80,80,80], [0,0,0])

        # lets display the rv_dir
        rv_dir_act = include_points(list(60*self.rv_dir), 1, 15, (1, 0 ,1))

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.AddActor(self.meshActor)

        # for plAct in [item for sublist in plane_storer for item in sublist]: # flatten list
        #     ren.AddActor(plAct)

        ren.AddActor(plane_storer[0][0]) # 4ch endo
        ren.AddActor(plane_storer[0][1]) # 4ch epi
        ren.AddActor(plane_storer[1][0]) # 2ch endo
        ren.AddActor(plane_storer[1][1]) # 2ch epi
        # ren.AddActor(plane_storer[2][0]) # 3ch endo
        # ren.AddActor(plane_storer[2][1]) # 3ch epi

        self.meshActor.GetProperty().SetOpacity(1.0)
        ren.AddActor(legend)
        ren.AddActor2D(textActor)
        ren.AddActor(axes)
        ren.AddActor(apexA)
        ren.AddActor(rv_dir_act)

        if disp:
            vtk_show(ren)

        return cut_poly_array, plane_storer, ren

    def set_orig_cut_poly_array(self, display_opt):
        self.orig_cut_poly_array, self.orig_planeActor, _ = self.get_cut_poly_array(self.original_planes,
                                                           self.orig_view_angles, display_opt, ['var'])

    def include_cut_poly_array(self, planes, fix_pts):
        """ function to obtain the actor for the slices
            only using data = self.endo_poly
        """
        planeActors = []

        for i in range(3):
            # get plane coefficients
            a = planes[i][0]
            b = planes[i][1]
            c = planes[i][2]

            # create vtk plane object
            VTKplane = vtk.vtkPlane()
            VTKplane.SetNormal(a, b, c)
            if fix_pts[0] == 'var':  # for variability test
                VTKplane.SetOrigin(self.epi_apex_node)
            else:  # for foreshortening test
                VTKplane.SetOrigin(fix_pts[1+i])

            # create cutter
            cutEdges = vtk.vtkCutter()
            cutEdges.SetInputData(self.endo_poly) # always cut through endo
            cutEdges.SetCutFunction(VTKplane)
            cutEdges.GenerateCutScalarsOn()
            cutEdges.GenerateTrianglesOn()
            cutEdges.SetValue(0, 0.5)

            # create strips # just for output purposes
            cutStrips = vtk.vtkStripper()
            cutStrips.SetInputConnection(cutEdges.GetOutputPort())
            cutStrips.Update()

            # get polydata from strips (just for output purposes)
            cutPoly = vtk.vtkPolyData()
            cutPts = cutStrips.GetOutput().GetPoints()
            cutPoly.SetPoints(cutPts)
            cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

            cutterMapper = vtk.vtkPolyDataMapper()
            cutterMapper.SetInputConnection(cutEdges.GetOutputPort())
            cutterMapper.ScalarVisibilityOff()

            # create plane actor
            planeActor = vtk.vtkActor()
            planeActor.SetMapper(cutterMapper)
            planeActor.GetProperty().SetColor(self.plane_colors[i])
            planeActor.GetProperty().SetLineWidth(6)

            # store the actors of the specific planes to add later into 1 renderer
            planeActors.append(planeActor)

        return planeActors

    def find_top_low_pts3(self, polydata_endo_epi):
        """
        This function finds the lowest point for each view by using the recommendations
        of the ACE (American Committee of Echocardiography):

        https://www.asecho.org/wp-content/uploads/2015/01/ChamberQuantification2015.pdf

        "At the mitral valve level, the contour is closed by connecting the two opposite sections
        of the mitral ring with a straight line. LV length is defined as the distance between the bisector
        of this line and the apical point of the LV contour, which is most distant to it. The use of the
        longer LV length between the apical two- and four-chamber views is recommended."
        """

        # 1. sort the points of the cut slice in order to easily find top points
        sorted_pts_endo_epi = self.sort_cutpoly_points(polydata_endo_epi)

        # 2. top points are easily found as ..
        top_point_1 = sorted_pts_endo_epi[0][0]
        top_point_2 = sorted_pts_endo_epi[0][-1]
        top_points = [top_point_1, top_point_2]

        # 3. lowest point is found by finding furthest distant point from the middle point
        middle_point = (top_point_1 + top_point_2)/2.0
        dists = scipy.spatial.distance.cdist(sorted_pts_endo_epi[0],middle_point.reshape((1,3)))
        lowest_pt_idx = np.argmax(dists)
        lowest_pt = sorted_pts_endo_epi[0][lowest_pt_idx]

        # make function to display top points and lowest point
        display_special_pts = 0
        if display_special_pts:
            sepi = include_points(list(sorted_pts_endo_epi[1]),
                        len(list(sorted_pts_endo_epi[1])), 7, (0,1,0))
            sendo = include_points(list(sorted_pts_endo_epi[0]),
                        len(list(sorted_pts_endo_epi[0])), 7, (0,1,0))
            a1 = include_points(list(lowest_pt), 1, 15, (1,0,0))
            a2 = include_points(list(top_point_1), 1, 15, (1,0,0))
            a3 = include_points(list(top_point_2), 1, 15, (1,0,0))
            ren = vtk.vtkRenderer()
            ren.AddActor(sendo)
            ren.AddActor(a1)
            ren.AddActor(sepi)
            ren.AddActor(a2)
            ren.AddActor(a3)
            vtk_show(ren)

        return lowest_pt, lowest_pt_idx, top_points, sorted_pts_endo_epi

    def sort_cutpoly_by_angle(self, polydata, cells_list, num_cells, numPoints):
        """
        Sorts cells list by clockwise angle.
        1) Convert to 2D
        2) Find center of the polycut curve
        3) From there, compute angle from center to every point.
        4) Order by angle!

        N.B. The meshes are aligned with PCA.
        The vertical direction is set as the line from the center_pt to the top point 1.
        We this vertical direction as a means to measure our angle FROM.
        Since project_to_xy_plane rotates original 3d coordinate system to another 2d one,
        we want to project the vertical point too.

        Returns:
            sorted_pts : ndarray (num_points x 3)
            sorted_idxs : ndarray (num_points x 1)
        """
        # 1: get numpy array of points from vtk polydata object
        points = polydata.GetPoints()
        pts = np.zeros((numPoints, 3), dtype=float)

        index = 0
        for cell in cells_list:
            for id in cell:
                pts[index] = np.asarray(points.GetPoint(id))
                index += 1

        # 2: convert them to 2d points and obtain the R rotation matrix
        pts_2d = project_onto_xy_plane(pts)

        # 3: compute center (average of all points)
        center_pt = np.mean(pts_2d, axis=0)

        # 4: find top points by pointdata label

        # 4: compute angles from center to average cell pts:nt        vertical_dir = vert_pt_2d - center_pt  # get vertical direction vector (from center pt to tp1)
        signed_angles = np.zeros((numPoints,), dtype=float)

        for i in range(numPoints):
            current_vec = pts_2d[i] - center_pt
            signed_angles[i] = compute_angle_between(vertical_dir, current_vec)
            self.pers_var = 1
            if self.pers_var: # ctrl-w for exit window key
                plt.figure()
                plt.scatter(center_pt[0], center_pt[1], color='r', s=2)
                plt.scatter(pts_2d[:,0], pts_2d[:,1], color='b', s=0.5)
                plt.scatter(vert_pt_2d[0], vert_pt_2d[1], color='g', s=10)
                plt.scatter(pts_2d[i][0], pts_2d[i][1], color='k', s=7)
                plt.xlabel('angle = ' + str(signed_angles[i]))
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.show()

        # 5: sort angles (no matter ascending or descending)
        sorted_idxs = np.argsort(signed_angles)

        # 6: sorted points
        sorted_pts = pts[sorted_idxs]

        return sorted_pts, sorted_idxs

    def sort_cutpoly_points(self, polydata_endo_epi):
        """
        Returns the cut polydatas points in order.
        Top points are the first and last points.
        Inputs:
            polydata_endo_epi : the polydata defining the slice (vtk polydata object)
                for both epi and endo.
        Returns:
            sorted_pts : the vtk polydata points sorted in order (ndarray)
        """
        sorted_endo_epi_cut_pts = []

        for polydata in polydata_endo_epi:
            polypoints = polydata.GetPoints() # get lowest point using octree point locator
            numPoints = polydata.GetNumberOfPoints()
            num_cells = polydata.GetNumberOfCells()
            np_polypts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

            if (num_cells == 1):
                cells_list = get_indices_from_cell_list(polydata, num_cells)
                sorted_idxs = cells_list
                sorted_pts = np_polypts[sorted_idxs]

            else:
                cells_list = get_indices_from_cell_list(polydata, num_cells)
                sorted_pts, sorted_idxs = self.sort_cutpoly_by_angle(polydata, cells_list, num_cells, numPoints)

            sorted_endo_epi_cut_pts.append(sorted_pts)

        return sorted_endo_epi_cut_pts

    def set_numSamples(self, numSamples):
        self.numSamples = numSamples

    def get_horizontal_points_interp(self, sorted_cut_endo_pts, lowest_pt_idx, display_opt):
        """
        Function gets the horizontal points using vtkIntersectLine function.

        This one is used for volume calculations.

        Ensure that transverse lines are perpendicular to the long axis.

        IMPORTANT: since the topline goes from top_pts[0] to top_pts[1], the first point to
        intersect (pointid1), corresponds to the points in the side of top_pts[0]

        N.B. Ensure that the left and right points == self.numSamples !
        """
        # assign special pts
        top_left = np.asarray(sorted_cut_endo_pts[0])
        top_right = np.asarray(sorted_cut_endo_pts[-1])
        low_pt = np.asarray(sorted_cut_endo_pts[lowest_pt_idx])


        # make polydata out of sorted endo pts
        numPoints = sorted_cut_endo_pts.shape[0]
        vtk_float_arr = numpy_support.numpy_to_vtk(num_array=np.asarray(sorted_cut_endo_pts), deep=True, array_type=vtk.VTK_FLOAT)
        vtkpts = vtk.vtkPoints()
        vtkpts.SetData(vtk_float_arr)

        # add the basal points that connect from top_right to top_left
        basal_pts = getEquidistantPoints(top_left, top_right, 100, 1, 1)
        num_basal_pts = len(basal_pts)
        for basal_pt in basal_pts:
            vtkpts.InsertNextPoint(basal_pt)

        # now make lines
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(numPoints + num_basal_pts)

        for i in range(numPoints + num_basal_pts):
            polyLine.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)

        # add points and lines to polydata container
        cut_endo_poly = vtk.vtkPolyData()
        cut_endo_poly.SetPoints(vtkpts)
        cut_endo_poly.SetLines(cells)

        # create tree for intersection process
        bspTree = vtk.vtkModifiedBSPTree() # bsp tree is much faster than obbtree due to rejection test
        bspTree.SetDataSet(cut_endo_poly)
        bspTree.BuildLocator()

        # get horizontal direction (cross product between vertical and normal)
        top_center = (top_left + top_right) / 2.0
        vertical_vec = normalize(top_center - low_pt)
        normal = find_plane_eq(top_left, top_right, low_pt)[:3]
        horizontal_vec = normalize(np.cross(normalize(normal), vertical_vec))

        # add distance on both sides to make sure the line can pass through the entire LV horizontally
        dist = np.linalg.norm(top_right - top_left)
        pSource_0 = low_pt + dist*horizontal_vec
        pTarget_0 = low_pt - dist*horizontal_vec

        # determine the length to travel from top to bottom
        max_dist = np.linalg.norm(low_pt - top_center)

        left_pts = []
        right_pts = []

        weights = np.linspace(0.02, 0.99, self.numSamples)

        for i in range(self.numSamples):
            # determine source and target points
            pSource = pSource_0 + weights[i]*max_dist*vertical_vec
            pTarget = pTarget_0 + weights[i]*max_dist*vertical_vec
            center = (pSource + pTarget) / 2.0

            # set empty variables
            subId = vtk.mutable(0)
            pcoords = [0, 0, 0]
            t = vtk.mutable(0)
            left = [0, 0, 0]
            right = [0, 0, 0]

            # # run interesect command
            # pointid1 = bspTree.IntersectWithLine(pSource, pTarget, 0.001, t, left, pcoords, subId)
            # pointid2 = bspTree.IntersectWithLine(pTarget, pSource, 0.001, t, right, pcoords, subId)

            # intersect with line that goes from source to center    or    target to center
            pointid1 = bspTree.IntersectWithLine(pSource, center, 0.001, t, left, pcoords, subId)
            pointid2 = bspTree.IntersectWithLine(pTarget, center, 0.001, t, right, pcoords, subId)

            if display_opt:
                rightact = include_points(list(top_right), 1, 10, (0,1,0))
                leftact = include_points(list(top_left), 1, 10, (0,1,0))
                lowptact = include_points(list(low_pt), 1, 10, (0,1,0))
                psourceact = include_points(list(pSource), 1, 10, (1,0,0))
                ptargetact= include_points(list(pTarget), 1, 10, (1,0,0))
                left_found = include_points(list(left), 1, 10, (1,1,0))
                right_found = include_points(list(right), 1, 10, (1,1,0))

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(cut_endo_poly)
                act = vtk.vtkActor()
                act.SetMapper(mapper)
                ren = vtk.vtkRenderer()
                ren.AddActor(act)
                ren.AddActor(psourceact)
                ren.AddActor(ptargetact)
                ren.AddActor(rightact)
                ren.AddActor(leftact)
                ren.AddActor(lowptact)
                ren.AddActor(left_found)
                ren.AddActor(right_found)
                vtk_show(ren)


            if pointid1 + pointid2 == 2: # i.e. pointid = 1 and pointid2 = 1
                left_pts.append(list(left))
                right_pts.append(list(right))

        #           display purposes       #
        # 1.a actor for all left and right pts
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(cut_endo_poly)

        all_act = vtk.vtkActor()
        all_act.SetMapper(mapper)

        left_act = include_points(left_pts, len(left_pts), 4, (1,0,0))
        right_act = include_points(right_pts, len(right_pts), 4, (1,0,0))
        low_pt_act = include_points(list(low_pt), 1, 10, (1,0,1))

        # 2.a now add horizontal lines
        VTK_horiz_all = vtk.vtkPoints()

        for i in range(len(left_pts)):
            VTK_horiz_all.InsertNextPoint(left_pts[i])
            VTK_horiz_all.InsertNextPoint(right_pts[i])

        lineArray = vtk.vtkCellArray()

        for i in range(len(left_pts)):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i + i) # 0, 2, 4, 6 ,..
            line.GetPointIds().SetId(1, i + i + 1) # 1, 3, 5, 7,...
            lineArray.InsertNextCell(line)

        # 2.b create polydata
        polyLine = vtk.vtkPolyData()

        # 2.c add points and lines to polydata container
        polyLine.SetPoints(VTK_horiz_all)
        polyLine.SetLines(lineArray)

        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(polyLine)

        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetColor(0, 0, 1)
        lineActor.GetProperty().SetLineWidth(2)

        # 3.a also add one more (line to represent vertical direction, to show perpendicular)
        long_axis_array = vtk.vtkCellArray()
        long_axis = vtk.vtkLine()
        long_axis.GetPointIds().SetId(0, 0)
        long_axis.GetPointIds().SetId(1, 1)
        long_axis_array.InsertNextCell(long_axis)

        long_axis_pts = vtk.vtkPoints()
        long_axis_pts.InsertNextPoint(top_center)
        long_axis_pts.InsertNextPoint(low_pt)

        # 3.b create actor for long axis line
        long_axis_polydata = vtk.vtkPolyData()

        # 2.c add points and lines to polydata container
        long_axis_polydata.SetPoints(long_axis_pts)
        long_axis_polydata.SetLines(long_axis_array)

        la_mapper = vtk.vtkPolyDataMapper()
        la_mapper.SetInputData(long_axis_polydata)

        la_act = vtk.vtkActor()
        la_act.SetMapper(la_mapper)
        la_act.GetProperty().SetColor(0, 0, 1)
        la_act.GetProperty().SetLineWidth(2)

        ren = vtk.vtkRenderer()
        ren.AddActor(all_act)
        ren.AddActor(right_act)
        ren.AddActor(left_act)
        ren.AddActor(low_pt_act)
        ren.AddActor(lineActor)
        ren.AddActor(la_act)

        if display_opt:
            vtk_show(ren)

        # ensure that left and right points have the same number of points as numSamples
        if len(left_pts) != self.numSamples or len(right_pts) != self.numSamples:
            print('Either left or right points do not have the same number of points as numSamples!')

        return left_pts, right_pts, ren

    def get_landmarks(self, sorted_cut_endo_pts, lowest_pt_idx, display_opt):
        """
        Function to get landmarks for PCA + SVC statistical analysis on shapes.

        IMPORTANT: since the topline goes from top_pts[0] to top_pts[1], the first point to
        intersect (pointid1), corresponds to the points in the side of top_pts[0]

        N.B. Ensure that the left and right points == self.numSamples !
        """

        # make polydata out of sorted endo pts
        numPoints = sorted_cut_endo_pts.shape[0]
        vtk_float_arr = numpy_support.numpy_to_vtk(num_array=np.asarray(sorted_cut_endo_pts), deep=True, array_type=vtk.VTK_FLOAT)
        vtkpts = vtk.vtkPoints()
        vtkpts.SetData(vtk_float_arr)
        cut_endo_poly = vtk.vtkPolyData()
        cut_endo_poly.SetPoints(vtkpts)

        # now make lines
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(numPoints)

        for i in range(numPoints):
            polyLine.GetPointIds().SetId(i, i) # from 0,1 then 2,3 then 4,5 ...

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)

        # add points and lines to polydata container
        cut_endo_poly.SetLines(cells)

        # create tree for intersection process
        bspTree = vtk.vtkModifiedBSPTree() # bsp tree is much faster than obbtree due to rejection test
        bspTree.SetDataSet(cut_endo_poly)
        bspTree.BuildLocator()

        top_left = np.asarray(sorted_cut_endo_pts[0])
        top_right = np.asarray(sorted_cut_endo_pts[-1])
        low_pt = np.asarray(sorted_cut_endo_pts[lowest_pt_idx])

        # get direction of lines
        line_dir = normalize(top_right - top_left)  # top_pt[0] to top_pt[1]

        # add distance on both sides to make sure the line can pass through the entire LV horizontally
        dist = np.linalg.norm(top_right - top_left)
        pSource_0 = top_right + dist*line_dir
        pTarget_0 = top_left - dist*line_dir

        # determine the length to travel from top to bottom
        top_center = (top_right + top_left)/2.0
        midline = normalize(low_pt - top_center)
        max_dist = np.linalg.norm(low_pt - top_center)

        left_pts = []
        right_pts = []

        weights = np.linspace(0.00, 0.98, self.numSamples)

        for i in range(self.numSamples):
            # determine source and target points
            pSource = pSource_0 + weights[i]*max_dist*midline
            pTarget = pTarget_0 + weights[i]*max_dist*midline
            center = (pSource + pTarget) / 2.0

            # set empty variables
            subId = vtk.mutable(0)
            pcoords = [0, 0, 0]
            t = vtk.mutable(0)
            left = [0, 0, 0]
            right = [0, 0, 0]

            # # run interesect command
            # pointid1 = bspTree.IntersectWithLine(pSource, pTarget, 0.001, t, left, pcoords, subId)
            # pointid2 = bspTree.IntersectWithLine(pTarget, pSource, 0.001, t, right, pcoords, subId)

            # intersect with line that goes from source to center    or    target to center
            pointid1 = bspTree.IntersectWithLine(pSource, center, 0.001, t, left, pcoords, subId)
            pointid2 = bspTree.IntersectWithLine(pTarget, center, 0.001, t, right, pcoords, subId)

            left_pts.append(list(left))
            right_pts.append(list(right))

        if display_opt:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cut_endo_poly)

            all_act = vtk.vtkActor()
            all_act.SetMapper(mapper)

            right_act = include_points(left_pts, len(left_pts), 4, (1,0,0))
            left_act = include_points(right_pts, len(right_pts), 4, (1,0,0))
            low_pt_act = include_points(list(low_pt), 1, 10, (1,0,1))

            top_right_act = include_points(list(top_right), 1, 10, (0,0,1))
            top_left_act = include_points(list(top_left), 1, 10, (0,0,1))

            ren = vtk.vtkRenderer()
            ren.AddActor(all_act)
            ren.AddActor(right_act)
            ren.AddActor(left_act)
            ren.AddActor(top_right_act)
            ren.AddActor(top_left_act)
            ren.AddActor(low_pt_act)

            vtk_show(ren)

        # ensure that left and right points have the same number of points as numSamples
        if len(left_pts) != self.numSamples or len(right_pts) != self.numSamples:
            print('Either left or right points do not have the same number of points as numSamples!')

        return left_pts, right_pts

    def compute_horizontal_distances(self, horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b):
        """
        Compute horizontal points using the left and right points for the 2ch and 4ch views
        """
        self.actual_num_samples = len(horiz_2ch_a)  # might be different to numSamples

        # horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b = self.equalize_horizontal_points(horiz_2ch_a, horiz_2ch_b,
        #                                                                                      horiz_4ch_a, horiz_4ch_b)

        horiz_2ch_dists = np.zeros((self.actual_num_samples, 1), dtype=float)
        horiz_4ch_dists = np.zeros((self.actual_num_samples, 1), dtype=float)

        for k in range(self.actual_num_samples):
            horiz_2ch_dists[k] = np.linalg.norm(np.array(horiz_2ch_a[k]) - np.array(horiz_2ch_b[k]))
            horiz_4ch_dists[k] = np.linalg.norm(np.array(horiz_4ch_a[k]) - np.array(horiz_4ch_b[k]))

        return horiz_2ch_dists, horiz_4ch_dists

    def equalize_horizontal_points(self, horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b):
        """
        Ensures that the left and right points have the same number of points
        Also ensures that there are the same number of points in both 2ch and 4ch (both equalling num_samples)
        In the case that they are not equal, this function removes the last point.
        """

        if (len(horiz_2ch_a) != len(horiz_2ch_b) or len(horiz_4ch_a) != len(horiz_4ch_b)):
            print('Left and right sides have different number of points!')

        if (len(horiz_2ch_a) != len(horiz_4ch_a)):
            print('Equalizing number of left and right points..')
            if (len(horiz_2ch_a) < len(horiz_4ch_a)):
                horiz_4ch_a = horiz_4ch_a[:len(horiz_2ch_a)]
                horiz_4ch_b = horiz_4ch_b[:len(horiz_2ch_a)]
                self.actual_num_samples = len(horiz_2ch_a)
            else:
                horiz_2ch_a = horiz_2ch_a[:len(horiz_4ch_a)]
                horiz_2ch_b = horiz_2ch_b[:len(horiz_4ch_a)]
                self.actual_num_samples = len(horiz_4ch_a)

        return horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b

    #  computing volumes ..

    def Simpson_bp(self, horiz_2ch_dists, horiz_4ch_dists, L):
        """
        Function computes LV volume using Simpson's modified formula.

        INPUTS:
            horiz_2ch are the horizontal distances in the 2 chamber view
            horiz_4ch are the horizontal distances in the 4 chamber view
            L should be the length of the left ventricular cavity!

        The height of each disc is calculated as a fraction (usually one-twentieth)
        of the LV long axis based on the longer of the two lengths from the two and four-chamber views
        See Figure 3 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2064911/
        """

        summ = 0
        for i in range(self.actual_num_samples): # starts at the top
            summ = summ + horiz_2ch_dists[i]*horiz_4ch_dists[i]

        const = (np.pi / 4.0) * (L / float(self.actual_num_samples))
        vol = const*summ

        return vol

    def set_ideal_volume(self, display_opt):
        """
        Get volume of original angles (60-140), C is the center of the rim!
        """
        # lowest_point_2ch, lowest_point_2ch_idx, highest_points_2ch, sorted_pts_endo_epi = self.find_top_low_pts3([self.orig_cut_poly_array[1][0],self.orig_cut_poly_array[1][1]])
        # horiz_2ch_a, horiz_2ch_b, _ = self.get_horizontal_points_interp(sorted_pts_endo_epi[0], lowest_point_2ch_idx, 0)

        lowest_point_4ch, lowest_point_4ch_idx, highest_points_4ch, sorted_pts_endo_epi = self.find_top_low_pts3([self.orig_cut_poly_array[0][0],self.orig_cut_poly_array[0][1]])
        horiz_4ch_a, horiz_4ch_b, _ = self.get_horizontal_points_interp(sorted_pts_endo_epi[0], lowest_point_4ch_idx, 0)

        # compute euclidean distances for 2ch and 4ch
        horiz_2ch_dists, horiz_4ch_dists = self.compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b,
                                                                             horiz_4ch_a, horiz_4ch_b)

        L = get_length_of_LV_cavity(horiz_2ch_a, horiz_2ch_b,
                                    horiz_4ch_a, horiz_4ch_b,
                                    lowest_point_2ch, lowest_point_4ch)

        ideal_vol = self.Simpson_bp(horiz_2ch_dists, horiz_4ch_dists, L)  # should be approximately 142 mL

        self.ideal_vol = ideal_vol
        self.ideal_horiz_2ch_a = horiz_2ch_a
        self.ideal_horiz_2ch_b = horiz_2ch_b
        self.ideal_horiz_4ch_a = horiz_4ch_a
        self.ideal_horiz_4ch_b = horiz_4ch_b
        self.ideal_lowest_point_2ch = lowest_point_2ch
        self.ideal_lowest_point_4ch = lowest_point_4ch

    #   orientation tests ..

    def orientation_test(self, angs_range, display_opt):
        print('starting orientation test ..')

        fix_to_apex = 1

        # set up offset range for 2 and 4ch views ..
        tch_angle_range = angs_range[0]
        fch_angle_range = angs_range[1]

        numDegrees = len(tch_angle_range)
        LV_vols_matrix = np.zeros((numDegrees, numDegrees), dtype=float)

        for (i, degree2) in enumerate(tch_angle_range):
            for (j, degree4) in enumerate(fch_angle_range):
                print('i, j = ', i, ' ', j)

                # get offseted 2ch view ...
                new_P = my_rodriguez_rotation(self.plane_pts,
                        self.axis_of_rot_normalized, math.radians(degree2))
                two_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

                # get offseted 4ch view ..
                new_P = my_rodriguez_rotation(self.plane_pts,
                        self.axis_of_rot_normalized, math.radians(degree4))
                four_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

                # set planes ..
                planes = np.vstack((four_ch_view_plane_normal,
                                    two_ch_view_plane_normal,
                                    self.original_planes[2])) # 4, 2, 3

                # get poly data arrays in order 4,2,3 ..
                cut_poly_array, _, ren_mesh = self.get_cut_poly_array(planes, [degree4, degree2],
                                                                0, ['var'])

                # # get horizontal points for 2ch view ..
                # if i == 12 and j==0:
                #     self.pers_var = 1
                lowest_point_2ch, lowest_point_2ch_idx, highest_points_2ch, sorted_pts_endo_epi = self.find_top_low_pts3([cut_poly_array[1][0],cut_poly_array[1][1]])
                horiz_2ch_a, horiz_2ch_b, ren_2ch = self.get_horizontal_points_interp(sorted_pts_endo_epi[0], lowest_point_2ch_idx, 0)

                # get horizontal points for 4ch view ..
                lowest_point_4ch, lowest_point_4ch_idx, highest_points_4ch, sorted_pts_endo_epi = self.find_top_low_pts3([cut_poly_array[0][0],cut_poly_array[0][1]])
                horiz_4ch_a, horiz_4ch_b, ren_4ch = self.get_horizontal_points_interp(sorted_pts_endo_epi[0], lowest_point_4ch_idx, 0)

                # display both mesh and views with multiple renderers
                if display_opt:
                    display_mesh_and_views(ren_mesh, ren_2ch, ren_4ch)

                # compute euclidean distances for 2ch and 4ch ..
                horiz_2ch_dists, horiz_4ch_dists = self.compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b,
                                                                                     horiz_4ch_a, horiz_4ch_b)

                L = get_length_of_LV_cavity(horiz_2ch_a, horiz_2ch_b,
                                            horiz_4ch_a, horiz_4ch_b,
                                            lowest_point_2ch, lowest_point_4ch)

                # assign data_array ..
                vol = self.Simpson_bp(horiz_2ch_dists, horiz_4ch_dists, L)  # should be approximately 142 mL
                LV_vols_matrix[i, j] = vol

        return LV_vols_matrix

    def foreshortening_test(self, ring_points, display_opt):
        print('starting foreshortening test ..')

        # Since we know direction of normal to the plane (given by the coefficnets a b c), we just need to do the following :
        #  offset_pt = apex_node + degree*normalized[a,b,c]
        num_ring_pts = len(ring_points)

        # declare array for storage ..
        LV_vols_matrix = np.zeros((num_ring_pts, num_ring_pts), dtype=float)

        # find top and low pts for the 2ch and 4ch views at the usual angles
        _, _, highest_4ch_orig, _ = self.find_top_low_pts3([self.orig_cut_poly_array[0][0],self.orig_cut_poly_array[0][1]])
        _, _, highest_2ch_orig, _ = self.find_top_low_pts3([self.orig_cut_poly_array[1][0],self.orig_cut_poly_array[1][1]])

        for (i, offset_2ch) in enumerate(ring_points):
            offset_pt_2ch = np.asarray(offset_2ch)

            for (j, offset_4ch) in enumerate(ring_points):
                offset_pt_4ch = np.asarray(offset_4ch)

                print('i, j = ', i, ' ', j)

                # get new offsetted 2 and 4 chamber views
                three_ch_view_plane_normal = self.original_planes[2]
                two_ch_view_plane_normal = find_plane_eq(offset_pt_2ch, highest_2ch_orig[0], highest_2ch_orig[1])
                four_ch_view_plane_normal = find_plane_eq(offset_pt_4ch, highest_4ch_orig[0], highest_4ch_orig[1])

                planes = np.vstack((four_ch_view_plane_normal,
                                    two_ch_view_plane_normal,
                                    three_ch_view_plane_normal))

                cut_poly_array, _, ren_mesh = self.get_cut_poly_array(planes,[j,i], 0,
                                                         ['foreshortening', offset_pt_4ch,
                                                         offset_pt_2ch, self.epi_apex_node])

                # 2 chamber
                # if i==0 and j==4:
                    # lowest_point_2ch, lowest_point_2ch_idx, highest_points_2ch, sorted_pts_endo_epi = self.find_top_low_pts3([cut_poly_array[1][0], cut_poly_array[1][1]])
                    # horiz_2ch_a, horiz_2ch_b, ren_2ch = self.get_horizontal_points_interp(sorted_pts_endo_epi[0],
                    #                                     lowest_point_2ch_idx, 1)

                # # 4 chamber
                if i==0 and j==4:
                    lowest_point_4ch, lowest_point_4ch_idx, highest_points_4ch, sorted_pts_endo_epi = self.find_top_low_pts3([cut_poly_array[0][0], cut_poly_array[0][1]])
                    horiz_4ch_a, horiz_4ch_b, ren_4ch = self.get_horizontal_points_interp(sorted_pts_endo_epi[0],
                                                        lowest_point_4ch_idx, 1)

                # # display both mesh and views with multiple renderers
                # if display_opt:
                #     display_mesh_and_views(ren_mesh, ren_2ch, ren_4ch)
                #
                # # compute euclidean distances for 2ch and 4ch
                # horiz_2ch_dists, horiz_4ch_dists = self.compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b,
                #                                                                      horiz_4ch_a, horiz_4ch_b)
                #
                # # set L = length of LV cavity (choose longest from 2ch or 4ch)
                # L = get_length_of_LV_cavity(horiz_2ch_a, horiz_2ch_b,
                #                             horiz_4ch_a, horiz_4ch_b,
                #                             lowest_point_2ch, lowest_point_4ch)
                #
                # # compute volume and errors
                # vol = self.Simpson_bp(horiz_2ch_dists, horiz_4ch_dists, L)  # should be approximately 142 mL
                # LV_vols_matrix[i, j] = vol

        return LV_vols_matrix


    def compute_ring_points(self, levels, pts_per_ring, fac, display_opt):

        self.meshActor.GetProperty().SetOpacity(1.0)
        apexA = include_points(list(self.epi_apex_node), 1, 15, (0,0,0))

        samas = []
        output_points = np.zeros((np.sum(pts_per_ring), 3), dtype=float)
        assembly = vtk.vtkAssembly()
        spheres = vtk.vtkAssembly()
        assembly_circles = vtk.vtkAssembly()

        if (len(pts_per_ring) != levels):
            print('check number of levels and weight')

        vl = np.linalg.norm(self.C - self.epi_apex_node)

        we = fac*vl

        for i in range(levels):
            radius = int((i+1)*we)
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(self.epi_apex_node)
            sphere.SetRadius(radius)

            d3 = vtk.vtkDelaunay3D()
            d3.SetInputData(self.epi_poly)
            d3.SetAlpha(0)
            d3.Update()

            surf = vtk.vtkDataSetSurfaceFilter()
            surf.SetInputConnection(d3.GetOutputPort())
            surf.Update()

            bo = vtk.vtkIntersectionPolyDataFilter()
            bo.SetInputConnection(0, sphere.GetOutputPort())
            bo.SetInputConnection(1, surf.GetOutputPort())
            bo.Update()

            # remove duplicate points from bo polydata
            cpd = vtk.vtkCleanPolyData()
            cpd.SetInputConnection(bo.GetOutputPort())
            cpd.Update()

            kdTree = vtk.vtkKdTreePointLocator()
            kdTree.SetDataSet(cpd.GetOutput())  # cut through endo polydata (not mesh)
            kdTree.BuildLocator()

            # sort the points (in circular motion)
            bo_pts = cpd.GetOutput().GetPoints()

            # sort circle points in clockwise manner
            ordered_pts, sorted_angles = sort_circular_pts(bo_pts)
            num_bo_pts = ordered_pts.shape[0]

            # subsample
            subsampled = equidistant_points(ordered_pts, sorted_angles, pts_per_ring[i])

            # store results in output_points
            cumsum = np.cumsum(pts_per_ring)
            cumsum = np.insert(cumsum, 0, 0)
            output_points[cumsum[i]:cumsum[i+1]] = subsampled

            # for display
            subsampledA = include_points(subsampled, subsampled.shape[0], 5, (1,1,0))
            assembly.AddPart(subsampledA)

        # # dont forget to also add the apex_node at the end
        # output_points[-1] = self.epi_apex_node

        if display_opt:
            ren = vtk.vtkRenderer()
            ren.SetBackground(1.0,1.0,1.0)
            ren.AddActor(self.meshActor)
#             ren.AddActor(subsampledA)
            ren.AddActor(apexA)
            ren.AddActor(assembly) # the subsampled points
            # ren.AddActor(spheres)
            # ren.AddActor(assembly_circles) # the intersections
            vtk_show(ren)

        return output_points

    def create_str_ring_offsets_labels(self, ring_points, pts_per_ring):
        circle_str = []
        for i in range(len(pts_per_ring)):
            for j in range(pts_per_ring[i]):
                st = 'c' + str(i) + '_' + str(j)
                circle_str.append(st)

        circle_str.append('apex_node')

        return circle_str

    def compute_parallel_points(self, sorted_pts, np, display_opt):
        """
        Computes the offset positions parallel to the current view, rather than using ring method.
        This method is a lot faster since it takes only offset points in one direction.
        Most of the motion in the view is along this view anyways, so it is a reasonable truncation to make.

        INPUTS:
            polypts : nd array of the cut (if you want to compute parallel points for 4ch, then
                        polypts should be the 2ch points).
            np : number of offset points.

        RETURNS:
            paralell_pts
        """
        # find lowest point (get point closest to epi_apex_node)
        numPoints = sorted_pts.shape[0]
        ds = np.zeros((numPoints,), dtype=float)

        epi_apex_node = np.asarray(self.epi_apex_node)

        for i in range(numPoints):
            ds[i] = np.linalg.norm(epi_apex_node - sorted_pts[i])

        lowest_point_id = np.argmin(ds)

        # create linspace using lowest point id
        num_pts_left = lowest_point_id
        num_pts_right = numPoints - lowest_point_id
        print('num pts left = ', num_pts_left)
        print('num points right = ', num_pts_right)

        scale = 0.3 # we only want 30% of points on one side
        max_ls_idx = int(scale * num_pts_left)
        max_rs_idx = int(scale * num_pts_right)
        idxs = np.arange(lowest_point_id-max_ls_idx, lowest_point_id+max_rs_idx, np, dtype=int)

        # get points
        parapts = np.zeros((np, 3), dtype=float)
        for i, idx in enumerate(idxs):
            parapts[i] = sorted_pts[idx]

        if display_opt:
            parapts_act = include_points(parapts, parapts.shape[0], 7, (0,1,0))
            ren = vtk.vtkRenderer()
            ren.SetBackground(1.0,1.0,1.0)
            ren.AddActor(self.endoActor)
            ren.AddActor(parapts_act)
            vtk_show(ren)

        return parapts


    #   specific angles/offsets testing ..

    def get_axes_of_plane(self, cut_poly_array, planes):

        lowest_point_2ch, _, highest_points_2ch, _ = self.find_top_low_pts3([cut_poly_array[1][0], cut_poly_array[1][1]])  # 2 ch
        lowest_point_4ch, _, highest_points_4ch, _ = self.find_top_low_pts3([cut_poly_array[0][0], cut_poly_array[0][1]])  # 4 ch

        z_axis_2ch = normalize(planes[1, :3])
        middle = (np.asarray(highest_points_2ch[0]) + np.asarray(highest_points_2ch[1])) / 2.0
        y_axis_2ch = normalize(middle - lowest_point_2ch)
        x_axis_2ch = -normalize(np.cross(z_axis_2ch, y_axis_2ch))

        z_axis_4ch = normalize(planes[0, :3])
        middle = (np.asarray(highest_points_4ch[0]) + np.asarray(highest_points_4ch[1])) / 2.0
        y_axis_4ch = normalize(middle - lowest_point_4ch)
        x_axis_4ch = -normalize(np.cross(z_axis_4ch, y_axis_4ch))

        axes_2ch = [x_axis_2ch, y_axis_2ch, z_axis_2ch]
        axes_4ch = [x_axis_4ch, y_axis_4ch, z_axis_4ch]

        return axes_2ch, axes_4ch

    def compute_ground_truth_volume(self, display_opt):
        """
        Computes ground truth volume
        Surface filter is used to get the surfaces from the outisde of the volume
        Note that output of d3 is a tetrahedral mesh!
        vtk mass properties only works for triangular mesh which is why we need this surface filter!

        the clean poly data is essential!!!!!!!! for volume calculation especially!!!!
        """

        self.meshActor.GetProperty().SetOpacity(0.2)
        self.meshActor.GetProperty().SetColor(1, 0, 0)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(self.endo_poly)

        d3 = vtk.vtkDelaunay3D()
        d3.SetInputConnection(clean.GetOutputPort())
        d3.SetTolerance(0.01)
        d3.SetAlpha(0.0)
        d3.Update()

        surfaceFilter = vtk.vtkDataSetSurfaceFilter()  # output is triangular mesh
        surfaceFilter.SetInputConnection(d3.GetOutputPort())
        surfaceFilter.Update()

        Mass = vtk.vtkMassProperties()
        Mass.SetInputConnection(surfaceFilter.GetOutputPort())
        Mass.Update()

        self.ground_truth_vol = Mass.GetVolume()/1000.0

        if display_opt:

            m = vtk.vtkDataSetMapper()
            m.SetInputConnection(d3.GetOutputPort())

            a = vtk.vtkActor()
            a.SetMapper(m)

            # set mapper for epi for visualization
            m2 = vtk.vtkDataSetMapper()
            m2.SetInputData(self.epi_poly)

            epi_actor = vtk.vtkActor()
            epi_actor.SetMapper(m2)
            epi_actor.GetProperty().SetOpacity(0.3)
            epi_actor.GetProperty().SetColor(1,0,0)

            ren = vtk.vtkRenderer()
            ren.SetBackground(0.0, 0.0, 0.0)
            ren.AddActor(epi_actor)
            ren.AddActor(a)

            vtk_show(ren)

    def compute_y_dists(self):
        """
        Computes y distance, defined as:
            offset point  to    epi_apex_node

        It must not be apex to lowest_pt of 2D view.
        """

        # declare arrays:
        dists_2ch = []
        dists_4ch = []
        dists_all = []

        for (i, offset_pt) in enumerate(ring_points):
            # compute distance for labels (simple euclidean dist) ..
            d = np.linalg.norm(np.asarray(offset_pt) - np.asarray(self.epi_apex_node))
            dists_2ch.append(d)
            dists_4ch.append(d)

        return dists_all

    def find_planeActor(self, cutEdges, text, color, display_opt):
        cutterMapper = vtk.vtkPolyDataMapper()
        cutterMapper.SetInputConnection(cutEdges.GetOutputPort())
        cutterMapper.ScalarVisibilityOff()

        # create plane actor ..
        planeActor = vtk.vtkActor()
        planeActor.SetMapper(cutterMapper)
        planeActor.GetProperty().SetColor(color)
        planeActor.GetProperty().SetLineWidth(6)

        if display_opt:
            # create text box to display the angles ..
            textActor = vtk.vtkTextActor()
            textActor.SetInput(text)
            textActor.SetPosition2(10, 40)
            textActor.GetTextProperty().SetFontSize(24)
            textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)

            ren = vtk.vtkRenderer()
            ren.AddActor(textActor)
            ren.AddActor(planeActor)
            ren.AddActor(self.meshActor)

            vtk_show(ren)

        return planeActor

    def get_edges_strips(self, plane, origin, text, color):
        """
        Computes vtkCutter and poly data from plane equation cutting
         through endo and epi.
        Returns:
            cut_poly_endo_epi : vtk polydata of slice for both endo and epi.
            planeActor_endo_epi : vtkActor of the slices for display.
        """
        datas = [self.endo_poly, self.epi_poly]
        cut_poly_endo_epi = [] # [0] is endo, [1] is epi
        planeActor_endo_epi = []

        for data in datas:
            a = plane[0]
            b = plane[1]
            c = plane[2]

            # create vtk plane object
            VTKplane = vtk.vtkPlane()
            VTKplane.SetNormal(a, b, c)
            VTKplane.SetOrigin(origin)

            # create cutter
            cutEdges = vtk.vtkCutter()
            cutEdges.SetInputData(data)
            cutEdges.SetCutFunction(VTKplane)
            cutEdges.GenerateCutScalarsOn()
            # cutEdges.GenerateTrianglesOn()
            cutEdges.SetValue(0, 0.5)

            # create strips # just for output purposes
            cutStrips = vtk.vtkStripper()
            # cutStrips.JoinContiguousSegmentsOn()
            cutStrips.SetInputConnection(cutEdges.GetOutputPort())
            cutStrips.Update()

            # # get polydata from strips (just for output purposes)
            cutPoly = vtk.vtkPolyData()
            cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
            cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

            # cutPoly = cutEdges.GetOutput()
            cut_poly_endo_epi.append(cutPoly)

            planeActor = self.find_planeActor(cutEdges, text, color, 0)
            planeActor_endo_epi.append(planeActor)

        return cut_poly_endo_epi, planeActor_endo_epi

    def store_horiz_points_clean(self, ring_points, display_opt):
        """
        Same as store_horiz_points but when determining landmarks, it sets 1 to views
        that lead to LV volume with perc error > 5% (wrt idealized Simpson volume, not ground truth)
        """
        plane_colors = [(0, 255, 0), (0, 0, 255), (0.9100, 0.4100, 0.1700)] #4,2,3

        num_ring_pts = len(ring_points)

        # declare arrays for storage ..
        landmarks = []
        landmarks_2chs = []
        landmarks_4chs = []

        y = []
        y_2ch = []
        y_4ch = []

        # find top and low pts for the 2ch and 4ch views at the ideal angles
        _, _, highest_2ch_orig, _ = self.find_top_low_pts3([self.orig_cut_poly_array[1][0],self.orig_cut_poly_array[1][1]])
        _, _, highest_4ch_orig, _ = self.find_top_low_pts3([self.orig_cut_poly_array[0][0],self.orig_cut_poly_array[0][1]])

        for (i, offset_2ch) in enumerate(ring_points):
            offset_pt_2ch = np.asarray(offset_2ch)

            two_ch_view_plane_normal = find_plane_eq(offset_pt_2ch, highest_2ch_orig[0], highest_2ch_orig[1])

            cutPoly_2ch, _ = self.get_edges_strips(two_ch_view_plane_normal, offset_pt_2ch,
                                                     "2ch ring_pt index = " + str(i), self.plane_colors[1])

            lowest_point_2ch, lowest_point_2ch_idx, highest_points_2ch, sorted_pts_endo_epi = self.find_top_low_pts3([cutPoly_2ch[0], cutPoly_2ch[1]])

            horiz_2ch_a, horiz_2ch_b = self.get_landmarks(sorted_pts_endo_epi[0], lowest_point_2ch_idx, 0)

            landmarks_3D = np.concatenate((horiz_2ch_a, np.flip(horiz_2ch_b, 0)))

            # remember to add the lowest point
            current_np = len(landmarks_3D)
            idx = int(np.floor(current_np/2.0)) # find middle index
            landmarks_3D = np.insert(landmarks_3D, idx, lowest_point_2ch, axis=0)
            landmarks_2D = project_onto_xy_plane(landmarks_3D)
            landmarks_2chs.append(landmarks_2D)

            # now assess volume for labels
            horiz_2ch_dists, horiz_4ch_dists = self.compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b,
                                                                                 self.ideal_horiz_4ch_a,
                                                                                 self.ideal_horiz_4ch_b)

            # set L = length of LV cavity (choose longest from 2ch or 4ch)
            L = get_length_of_LV_cavity(horiz_2ch_a, horiz_2ch_b,
                                        self.ideal_horiz_4ch_a, self.ideal_horiz_4ch_b,
                                        lowest_point_2ch, self.ideal_lowest_point_4ch)

            # compute percentage error
            vol = self.Simpson_bp(horiz_2ch_dists, horiz_4ch_dists, L)
            abs_error_vol = np.abs(vol - self.ideal_vol) # make all errors positive
            perc_err = 100.0*(abs_error_vol/self.ideal_vol)

            if perc_err > 5.0:
                y_2ch.append(int(1))
            else:
                y_2ch.append(int(0))

        landmarks.append(landmarks_2chs)
        y.append(y_2ch)

        for (i, offset_4ch) in enumerate(ring_points):
            offset_pt_4ch = np.asarray(offset_4ch)

            four_ch_view_plane_normal = find_plane_eq(offset_pt_4ch, highest_4ch_orig[0], highest_4ch_orig[1])

            cutPoly_4ch, _ = self.get_edges_strips(four_ch_view_plane_normal, offset_pt_4ch,
                                                     "4ch = " + str(i), plane_colors[0])

            lowest_point_4ch, lowest_point_4ch_idx, highest_points_4ch, sorted_pts_endo_epi = self.find_top_low_pts3([cutPoly_4ch[0], cutPoly_4ch[1]])

            horiz_4ch_a, horiz_4ch_b = self.get_landmarks(sorted_pts_endo_epi[0], lowest_point_4ch_idx, 0)

            landmarks_3D = np.concatenate((horiz_4ch_a, np.flip(horiz_4ch_b, 0)))

            # dont forget to add the type I landmarks (lowest points)
            current_np = len(landmarks_3D)
            idx = int(np.floor(current_np/2.0))
            landmarks_3D = np.insert(landmarks_3D, idx, lowest_point_4ch, axis=0)
            landmarks_2D = project_onto_xy_plane(landmarks_3D)
            landmarks_4chs.append(landmarks_2D)

            # now asssess volume for labels
            horiz_2ch_dists, horiz_4ch_dists = self.compute_horizontal_distances(self.ideal_horiz_2ch_a,
                                                                                 self.ideal_horiz_2ch_b,
                                                                                 horiz_4ch_a,
                                                                                 horiz_4ch_b)

            # set L = length of LV cavity (choose longest from 2ch or 4ch)
            L = get_length_of_LV_cavity(self.ideal_horiz_2ch_a, self.ideal_horiz_2ch_b,
                                        horiz_4ch_a, horiz_4ch_b,
                                        self.ideal_lowest_point_2ch, lowest_point_4ch)

            # compute percentage error
            vol = self.Simpson_bp(horiz_2ch_dists, horiz_4ch_dists, L)
            abs_error_vol = np.abs(vol - self.ideal_vol) # make all errors positive
            perc_err = 100.0*(abs_error_vol/self.ideal_vol)

            if perc_err > 5.0:
                y_4ch.append(int(1))
            else:
                y_4ch.append(int(0))

        landmarks.append(landmarks_4chs)
        y.append(y_4ch)

        return landmarks, y

    def compute_thickness(self):
        """
        Computes thickness at the rim level (not the whole endocardium).
        """
        com = vtk.vtkCenterOfMass()
        com.SetInputData(self.inner_rim_poly)
        center = np.asarray(com.GetCenter()) # take center from inner points (not outer)

        irp_numpy = numpy_support.vtk_to_numpy(self.inner_rim_poly.GetPoints().GetData())
        orp_numpy = numpy_support.vtk_to_numpy(self.outer_rim_poly.GetPoints().GetData())

        # compute average radius ..
        rs_inner = np.linalg.norm(irp_numpy - np.tile(center, (irp_numpy.shape[0], 1)), axis = 1)
        rs_outer = np.linalg.norm(orp_numpy - np.tile(center, (orp_numpy.shape[0], 1)), axis = 1)

        # average out
        r_inner = np.mean(rs_inner)
        r_outer = np.mean(rs_outer)

        # compute distance
        d = r_outer - r_inner
        self.thickness = d

        return d

    def find_epi_apex_point(self, display_opt):
        """
        finds apex point based on gaussian curvature (mean method) ..
        """
        curv = vtk.vtkCurvatures()
        curv.SetCurvatureTypeToMean()
#         curv.SetCurvatureTypeToMinimum()
        # curv.SetCurvatureTypeToMaximum()
#         curv.SetCurvatureTypeToGaussian()

        curv.SetInputData(self.epi_poly)
        curv.Update()

        curv_numpy = numpy_support.vtk_to_numpy(curv.GetOutput().GetPointData().GetScalars())
        max_curv_ptid = np.argmax(curv_numpy)
        max_curv_pt = self.epi_poly.GetPoints().GetPoint(max_curv_ptid)

        self.epi_apex_node = max_curv_pt
        pointActor = include_points(list(self.epi_apex_node), 1, 15, (0,0,0))

        sc_r = curv.GetOutput().GetScalarRange()

        # build lut
        scheme = 16
        colorSeries = vtk.vtkColorSeries()
        colorSeries.SetColorScheme(scheme)

        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()

        numColors = colorSeries.GetNumberOfColors()
        for i in range(numColors):
            color = colorSeries.GetColor(i)
            dColor = np.zeros((3,))
            dColor[0] = color[0]/255.0
            dColor[1] = color[1]/255.0
            dColor[2] = color[2]/255.0
            t = sc_r[0] + (sc_r[1] - sc_r[0]) / (numColors-1)*i
            lut.AddRGBPoint(t, dColor[0], dColor[1], dColor[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(curv.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(sc_r)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mapper.GetLookupTable())
        scalarBar.SetTitle('scalar bar')
        scalarBar.SetNumberOfLabels(5)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
#         ren.AddActor(self.meshActor)
        ren.AddActor(actor)
        ren.AddActor(pointActor)
        ren.AddActor2D(scalarBar)

        if display_opt:
            vtk_show(ren)

    def find_endo_apex_point(self, display_opt):
        """
        finds apex point based on gaussian curvature (mean method) ..
        """
        curv = vtk.vtkCurvatures()
#         curv.SetCurvatureTypeToMean()
        curv.SetCurvatureTypeToMinimum()

        curv.SetInputData(self.endo_poly)
        curv.Update()

        curv_numpy = numpy_support.vtk_to_numpy(curv.GetOutput().GetPointData().GetScalars())
        max_curv_ptid = np.argmax(curv_numpy)
        max_curv_pt = self.endo_poly.GetPoints().GetPoint(max_curv_ptid)

        self.endo_apex_node = max_curv_pt
        pointActor = include_points(list(self.endo_apex_node), 1, 15, (0,0,0))

        sc_r = curv.GetOutput().GetScalarRange()

        # build lut
        scheme = 16
        colorSeries = vtk.vtkColorSeries()
        colorSeries.SetColorScheme(scheme)

        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()

        numColors = colorSeries.GetNumberOfColors()
        for i in range(numColors):
            color = colorSeries.GetColor(i)
            dColor = np.zeros((3,))
            dColor[0] = color[0]/255.0
            dColor[1] = color[1]/255.0
            dColor[2] = color[2]/255.0
            t = sc_r[0] + (sc_r[1] - sc_r[0]) / (numColors-1)*i
            lut.AddRGBPoint(t, dColor[0], dColor[1], dColor[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(curv.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(sc_r)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mapper.GetLookupTable())
        scalarBar.SetTitle('scalar bar')
        scalarBar.SetNumberOfLabels(5)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        ren.AddActor(actor)
        ren.AddActor(pointActor)
        ren.AddActor2D(scalarBar)


        if display_opt:
            vtk_show(ren)

# non mesh class functions ..

def display_mesh_and_views(ren_mesh, ren_2ch, ren_4ch):
    """
    This function creates a 1x3 window that displays how the views are oriented
    (cutting through mesh) and the views themselves in 2D.
    """
    # 1. create multiple rendering views and viewports
    viewPorts = split_window(3,1)
    print('viewports = ', viewPorts)
    renderers = []

    # 2. first viewport: for mesh display with views
    ren_mesh.SetViewport(viewPorts[0,:])
    ren_mesh.SetBackground(1.0, 1.0, 1.0)
    renderers.append(ren_mesh)

    # 2. now display the second view port (2ch view)
    ren_2ch.SetViewport(viewPorts[1,:])
    textActor = vtk.vtkTextActor()
    textActor.SetInput("A2C")
    textActor.SetPosition2(10, 40)
    textActor.GetTextProperty().SetFontSize(24)
    textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
    ren_2ch.AddActor(textActor)
    renderers.append(ren_2ch)

    # 3. lastly display the third view port (4ch view)
    ren_4ch.SetViewport(viewPorts[2,:])
    textActor = vtk.vtkTextActor()
    textActor.SetInput("A4C")
    textActor.SetPosition2(10, 40)
    textActor.GetTextProperty().SetFontSize(24)
    textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
    ren_4ch.AddActor(textActor)
    renderers.append(ren_4ch)

    vtk_multiple_renderers(renderers, 800, 800)

### euqalizing inertial mass and maximizing vertical axis length

def criterion_2d_vertical(P2, P1, endo_epi_pts, weight, numSamples):
    """
    Function returns value of minimization function:
        Criterion1 = 1 / length_of_LV_axis
        Criterion2 = np.abs(D2 - D1 + D4 - D3)
            where D1, D2 = distance of endo points to axis
                  D3, D4 = distance of epi points to axis
        C = weight * C1  + (1-weight) * C2

    Inputs:
        P2 : lowest point of parabola
        P1 : mean basal point of inner rim poly
        weight : from minimization function
        endo_epi_pts = [endo_pts, epi_pts] : ndarray

    Note: P1 should be the first argument, so optimize.fmin
    knows that this is the argument of the minimization.
    """

    # criterion 1 : length of midline axis
    axis_line = P1 - P2
    C1 = 1.0 / np.linalg.norm(axis_line)

    # criterion 2 : distance to axis
    distances, _ = distances_to_axis(endo_epi_pts, P1, P2, numSamples)
    d1 = np.abs(np.sum(distances[0][distances[0]>0]))
    d2 = np.abs(np.sum(distances[0][distances[0]<=0]))
    d3 = np.abs(np.sum(distances[1][distances[1]>0]))
    d4 = np.abs(np.sum(distances[1][distances[1]<=0]))

    # crtierion 2 : equalizing inertial mass
    C2 = np.abs(d2-d1 + d4-d3)

    # join two criterias with weights
    C = weight*C1 + (1.0-weight)*C2

    return C

def distances_to_axis(endo_epi_pts, P1, P2, numSamples):
    """
    Function computes the distances to axis using the current solution P2 (lowest point).

    Steps:
        1) Rotate the contour such that they are upside down, such that all x>0
            is D1 and all x<0 is D2.
        2) Distances are computed from endo/epi points to the axis.


    Inputs:
        endo_epi_pts : [endo_pts, epi_pts] : ndarray

    Returns:
        distances : [d1, d2, d3, d4]
            where d1 = right side distances of endo points from axis
                  d2 = left side distances of endo points from axis
                  d3 = right side distances of epi points from axis
                  d4 = left side distances of epi points from axis

        endo_landmarks : actual landmarks then used for classification (using closest y value
            to the linspacing)
    """
    # find rotation matrix that rotates from axis to y-axis
    axis = P1 - P2
    R = build_rotation_matrix(np.array([0,1]), axis)

    # apply rotation matrix (rotate to y axis)
    rot_endo_pts = np.dot(R, endo_epi_pts[0].T).T
    rot_epi_pts = np.dot(R, endo_epi_pts[1].T).T

    # find distances of all endo epi pts to the y axis
    endo_epi_landmarks = []
    distances = [] # d1, d2, d3, d4

    for pts in [rot_endo_pts, rot_epi_pts]:
        # store distances D endo D epi
        distances.append(get_distances_to_y(pts))

    return distances, endo_epi_landmarks

def get_distances_to_y(pts):
    """
    Finds distances of pts to y axis in 2D.
    Includes positive and negative signs.
    pts : ndarray (n_shapes x 2)
    Returns distances (n_shapes)
    """

    y_axis = np.array([0,1])
    num_pts = pts.shape[0]
    distances = np.linalg.norm(pts - np.tile(y_axis, [num_pts, 1]), axis=1)

    # discriminate between left and right side points since norm always returns
    # positive distances .
    signs = np.zeros((pts.shape[0],1))
    for i, pt in enumerate(pts):
        signs[i] = sign_line(pt, [0,-10], [0,10])

    distances = signs * distances # elementwise mult

    return distances

def build_rotation_matrix(vec_new, vec_orig):
    """
    Builds an rotation matrix that rotates original coordinate axis (vec_orig)
    to a new coord axis (vec_new).

    Inputs:
        vec_orig : one of the axis in the original coordinate system
        vec_new : corresponding axis but in the new coordinat system
    """
    theta =  np.arccos(np.dot(vec_new, vec_orig) / (np.linalg.norm(vec_new) * np.linalg.norm(vec_orig)))

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return R

def sign_line(pt, P1, P2):
    """
    Function returns sign to determine which side of line L, point pt
    is located.
    """
    x1, y1 = P1
    x2, y2 = P2
    x, y = pt

    return np.sign((x - x1)*(y2 - y1) - (y-y1)*(x2-x1))

def split_LR(pts, axis):
    """
    Function array of points into left and right based
    on axis. It vectorizes the function 'sign_line'.
    """
    left_pts = []
    right_pts = []

    for i, pt in enumerate(pts):
        if sign_line(pt, P1, P2) > 0:
            left_pts.append(pt)
        else:
            right_pts.append(pt)

    return left_pts, right_pts

def compute_angle_between(a, b):
    """ Returns the angle in radians between vectors 'a' and 'b'::
    """
    ax, ay = a
    bx, by = b

    dot = np.dot(a,b)
    crx = ax*by - ay*bx
    scale = 180.0/np.pi # to convert from radians to degrees

    angle = np.arctan2(np.abs(crx), dot)*scale

    # make sure angle goes from 0 to 360
    if crx < 0.0:
        angle = 360.0 - angle

    return angle

## rotation

def rotate_2d_pts(pts, lowest_pt, top_points):
    """
        Rotates the 2d landmarks (which lie in the xy plane) and rotates
        them such that:
        - y is the view midline (see below)
        - x is perpendicular to y.

        Inputs:
            both lowest_pt, and top_points should be 2D too
        Should be ran after project_onto_xy_plane since that function
        only brings the 3d points to xy plane (doesnt align them once
        it is on the xy plane).

        view midline : lowest_point to center of top points
    """
    # calculate theta (angle between view midline) and y_axis
    top_points = np.asarray(top_points)
    lowest_pt = np.asarray(lowest_pt)
    center_top_pts = np.mean(top_points, axis=0)
    view_midline = np.array(center_top_pts - lowest_pt)
    a,b = view_midline
    x,y = np.array([0,1])
    theta = np.arctan2(-b*x + a*y , a*x + b*y) # signed angle

    # compute rotation matrix
    R = np.array([[np.cos(theta) , -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

    # apply rotation matrix
    rotated_pts = np.dot(R, pts)

    return rotated_pts

def get_line_actor(pts, lw, color):
    """
    Returns vtk line actor from numpy array pts.
    pts : (n_points x dim)
    """
    vtk_points = vtk.vtkPoints()
    num_points = int(pts.shape[0]/2)

    for pt in pts:
        vtk_points.InsertNextPoint(pt)

    lineArray = vtk.vtkCellArray()



    for i in range(num_points):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i + i)
        line.GetPointIds().SetId(1, i + i + 1)
        lineArray.InsertNextCell(line)

    # create polydata
    polyLine = vtk.vtkPolyData()

    # add points and lines to polydata container
    polyLine.SetPoints(vtk_points)
    polyLine.SetLines(lineArray)

    # create pipeline for display*
    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputData(polyLine)

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetColor(color)
    lineActor.GetProperty().SetLineWidth(lw)

    return lineActor

def project_vectors_ab(a, b):
    """
        Project vector a onto b.
        Both must be ndarray.

        Returns scalar projection! (not the projection vector).
    """
    # print('dot = ', np.dot(a,b))
    # print('norm = ', np.linalg.norm(b))
    return np.dot(a, b) / np.linalg.norm(b)

def find_minima_via_projections(line, arr, weight):
    """
        Function finds minimum point of points by
        ion all points
        in 'arr' onto line 'line'.

        Weight determines how long the line is.

        Returns index of point in arr that corresponds to minimum.
    """
    top_pt = weight*line
    low_pt = -weight*line
    x_line = top_pt - low_pt

    projs = np.zeros((arr.shape[0],), dtype=float)
    for i, pt in enumerate(arr):
        vec = pt - low_pt
        projs[i] = project_vectors_ab(vec, x_line)

    return np.argmin(projs)

def get_axes_actor(scales, translates):

    transform = vtk.vtkTransform()
    transform.Scale(scales[0],scales[1],scales[2])
    transform.Translate(translates[0], translates[1], translates[2])
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    return axes

def MakevtkPoints(points, deep=True):
    """ Convert numpy points to a vtkPoints object """

    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(numpy_support.numpy_to_vtk(points, deep=deep))
    return vtkpts

def equidistant_points(ordered_pts, sorted_angles, N):
    """
       Returns N equidistant points from circular points.
       If N=4, then finds angles in sorted_angles that are separated
                equally into 4.

       Input:
        ordered_pts : points sorted in clockwise order (num_pts x 3)
        sorted_angles : the corresponding angles for each point (in circle)
                        using ordered_pts[0] as the reference point. (num_pts x 1)

    """

    # simply find equally distant angles in sorted_angles
    # we assume that there are enough points in the circle, uniform distribution of points around circle

    num_pts = ordered_pts.shape[0]
    eq_pts = np.zeros((N,3), dtype=float)
    idx_jump = int(np.floor(float(num_pts)/N))
    eq_pts[0] = ordered_pts[0]

    current_idx = idx_jump
    for i in range(1,N):
        eq_pts[i] = ordered_pts[current_idx]
        current_idx += idx_jump

    return eq_pts

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]

def sort_circular_pts(vtk_points):
    """
        Sorts points in circle by calculating angle between reference line
        and angle lines.
        Returns numpy array of sorted points
            and the sorted (signed) angles.

        URL:
            https://gamedev.stackexchange.com/questions/69475/how-do-i-use-the-dot-product-to-get-an-angle-between-two-vectors
    """

    # convert vtk points to numpy
    cpd_numpy = numpy_support.vtk_to_numpy(vtk_points.GetData())
    circle_pts_2D = project_onto_xy_plane(cpd_numpy)
    num_bo_pts = circle_pts_2D.shape[0]

    # select reference point and reference line
    ref_pt = circle_pts_2D[0]
    com = np.mean(circle_pts_2D, axis=0)
    a,b = normalize(ref_pt - com)

    # compute angles
    signed_angles = np.zeros((num_bo_pts, ), dtype=float)
    for i, pt in enumerate(circle_pts_2D):
        x, y = normalize(pt - com)
        theta = np.arctan2(-b*x + a*y , a*x + b*y)
        # theta = np.arctan2(angle_line[1], angle_line[0]) - np.arctan2(ref_line[1], ref_line[0])
        signed_angles[i] = theta

    # sort angles
    sort_idxs = np.argsort(signed_angles)
    sorted_pts = cpd_numpy[sort_idxs]

    return sorted_pts, np.sort(signed_angles)

def set_polydata_to_sorted_numpy(polydata):
    """
    Sorts polydata (cut plane) points from one end to the other.
    Returns the sorted points. Deprecated.
    """

    polys = polydata.GetPolys() # get connectivity of polygon cells (polygon : {triangles, tetra, hexa, ..etc.})
    num_cells = polys.GetNumberOfCells()

    polypoints = polydata.GetPoints() # get lowest point using octree point locator
    # total_num_pts = polydata.GetNumberOfPoints()
    # print('total num pts = ', total_num_pts)

    polys.InitTraversal()

    sorted_pts = []

    for i in range(num_cells):
        ptIds = vtk.vtkIdList()
        polys.GetNextCell(ptIds)
        num_pts = ptIds.GetNumberOfIds()

        for j in range(num_pts):
            sorted_pts.append(polypoints.GetPoint(ptIds.GetId(j)))

    return np.asarray(sorted_pts)

def get_indices_from_cell_list(polydata, num_cells):
    """
    The cut_poly_array for each view is a concatenation of disconnected segments.
    This function returns the indices of the segments individually and stacks them
    into cells_list.

    e.g. cells_list = [pt indices for segment 1, pt indices for segment 2, ... ]

    """
    cells_list = []

    polys = polydata.GetPolys() # get connectivity of polygon cells (polygon : {triangles, tetra, hexa, ..etc.})
    polys.InitTraversal()  # initialize traversal so getnextcell starts at 0

    temp_list = []

    for i in range(num_cells):
        ptIds = vtk.vtkIdList()
        polys.GetNextCell(ptIds)
        num_pts = ptIds.GetNumberOfIds()
        cellids = []

        for j in range(num_pts):
            current_ptid = ptIds.GetId(j)
            if current_ptid in temp_list:
                continue
            else:
                cellids.append(current_ptid)

            temp_list.append(current_ptid)

        cells_list.append(cellids)


    return cells_list

def find_cell_intersections(num_cells, cells_list):
    """
    Finds the point ids that intersect for each segment.
    e.g. cell1 has 2 intersections at each end point
    We repeat for all cells and store them in cells_intersect_ids.
    """
    cells_intersect_ids = []

    for i in range(num_cells):
        inters_list = []
        for j in range(num_cells):
            if i != j:
                # count number of intersections
                inters = np.intersect1d(cells_list[i], cells_list[j])
                if (inters.size != 0):
                    inters_list.append(np.asscalar(inters))

        cells_intersect_ids.append(np.asarray(inters_list))



    return cells_intersect_ids

def get_length_of_LV_cavity(horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b, lowest_pt_2ch, lowest_pt_4ch):
    """
    Function that returns length of long axis L for Simpson's volume calculation.

    Recommendation by ACE:
    "The use of the longer LV length between the apical two- and four-chamber views is recommended."

    Note: L should be computed for every pair of 2ch+4ch views (not per mesh)
    """
    top_center_2ch = (np.asarray(horiz_2ch_a[0]) + np.asarray(horiz_2ch_b[0])) / 2.0
    L_2ch = np.linalg.norm(top_center_2ch - np.asarray(lowest_pt_2ch))/1000.0

    top_center_4ch = (np.asarray(horiz_4ch_a[0]) + np.asarray(horiz_4ch_b[0])) / 2.0
    L_4ch = np.linalg.norm(top_center_4ch - np.asarray(lowest_pt_4ch))/1000.0

    if L_2ch > L_4ch:
        L = L_2ch
    else:
        L = L_4ch

    return L

def get_factors(val):
    """
    Get two factors of val that are as close as possible
    """
    N = np.sqrt(val)
    N = np.floor(N)
    M = val/N

    while (val % N != 0):
        N = N-1
        M = val/N

    return int(M), int(N)

def display_as_pandas(arr, index, columns):
    pd_arr = pd.DataFrame(arr, index=index, columns=columns)
    display(pd_arr)

def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    # create renderer window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(600, 600)

    # create render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()

    # start the display
    window.Render()
    interactor.Start()

def include_points(points, num_points_to_add, pointsize, color):
    """ function to include points on actor for display
    Inputs:
        points (list [] or vtkpoints() object) : points to be added
        num_points
    """


    pointPoly = vtk.vtkPolyData()

    # if points is a list and not a vtkobject..
    if (type(points) is list or type(points) is np.ndarray):
        VTKpoints = vtk.vtkPoints()
        if (num_points_to_add > 1):     # if more than 1 points loop..
            for i in range(num_points_to_add):
                VTKpoints.InsertNextPoint(points[i])
        else:                           # if only 1 point...
            VTKpoints.InsertNextPoint(points)

        pointPoly.SetPoints(VTKpoints)
    else:  # if points is a vtk object just add like this..
        pointPoly.SetPoints(points)

    # add vertex at each point (so no need to insert vertices manually)
    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(pointPoly)

    pointM = vtk.vtkPolyDataMapper()
    pointM.SetInputConnection(vertexFilter.GetOutputPort())

    pointA = vtk.vtkActor()
    pointA.SetMapper(pointM)
    pointA.GetProperty().SetColor(color)
    pointA.GetProperty().SetPointSize(pointsize)

    return pointA

def getEquidistantPoints(p1, p2, n, first_inc, last_inc):
    """
    Generates n points between p1 and p2.
    first_inc : 1 to include first point
    last_inc : 1 to include last point
    """
    pts = np.column_stack((np.linspace(p1[0], p2[0], n+1),
                        np.linspace(p1[1], p2[1], n+1),
                        np.linspace(p1[2], p2[2], n+1)))

    if first_inc == 0:
        pts = pts[1:] # delete the first one
    if last_inc == 0:
        pts = pts[:-1] # delete the last one

    return pts

# fitting circle ..

def get_xyz_vecs(points, npts):  # points is a vtkPoints() object
    x = np.zeros((npts), dtype=float)
    y = np.zeros((npts), dtype=float)
    z = np.zeros((npts), dtype=float)

    for i in range(npts):
        x[i] = points.GetPoint(i)[0]
        y[i] = points.GetPoint(i)[1]
        z[i] = points.GetPoint(i)[2]

    return x, y, z

def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:, np.newaxis]*u + r*np.sin(t)[:, np.newaxis]*np.cross(n, u) + C
    return P_circle

def generate_circle_by_angles(t, C, r, theta, phi):
    """
    Orthonormal vectors n, u, <n,u>=0
    """
    n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    u = np.array([-np.sin(phi), np.cos(phi), 0])

    P_circle = r*np.cos(t)[:, np.newaxis]*u + r*np.sin(t)[:, np.newaxis]*np.cross(n, u) + C

    return P_circle

def fit_circle_2d(x, y, w=[]):
    """
    Function to fit circle in 2D.

    Find center [xc, yc] and radius r of circle fitting to set of 2D points
    Optionally specify weights for points

    Implicit circle function:
      (x-xc)^2 + (y-yc)^2 = r^2
      (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
      c[0]*x + c[1]*y + c[2] = x^2+y^2

    Solution by method of least squares:
      A*c = b, c' = argmin(||A*c - b||^2)
      A = [x y 1], b = [x^2+y^2]
    """

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b)[0]

    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r

def rodrigues_rot(P, n0, n1):
    """
    RODRIGUES ROTATION:
    Rotate given points based on a starting and ending vector
    Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))

    If P is only 1d array (coords of single point), fix it to be matrix
    """

    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k, P[i])*np.sin(theta) + \
            k*np.dot(k, P[i])*(1-np.cos(theta))

    return P_rot

def angle_between(u, v, n=None):
    """
        Get angle between vectors u,v with sign based on plane with unit normal n
    """
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))

def set_axes_equal_3d(ax):
    """
        Make axes of 3D plot to have equal scales
        This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
            which were not working for 3D
    """

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 0] - limits[:, 1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = 2
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig

# extraction of chamber views..

def my_rodriguez_rotation(P, k, theta):
    """
    P are the old points to be rotated
    k is the axis about which the points are rotated
    theta is the degree in radians
    """

    P_rot = np.zeros((len(P), 3))

    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k, P[i])*np.sin(theta) + \
            k*np.dot(k, P[i])*(1.0-np.cos(theta))

    return P_rot

def find_plane_eq(p1, p2, p3):
    """
    From three points (p1, p2, p3) in a plane, it returns the plane equation
    ax1 + bx2 + cx3 = d

    Note: normal of the plane is given by n = (a,b,c)
    """

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    plane_eq = np.array([a, b, c, d])

    return plane_eq

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# function to project onto 2D

def convert_3d_plane_to_2d(cut_numpy):

    aortic_pt = cut_numpy[0, :] # get first and last points of the view (corresponds to the point closest to the aortic and mitral side respectively)
    mitral_pt = cut_numpy[-1, :]

    # set origin as the point in the center (near the apex)
    idx = int(np.floor(len(cut_numpy)/2))
    apex_pt = cut_numpy[idx, :]

    N = cut_numpy.shape[0]
    localz = np.cross((aortic_pt - apex_pt), (mitral_pt - apex_pt))
    unitz = localz / np.linalg.norm(localz, 2)
    localx = aortic_pt - apex_pt
    unitx = localx / np.linalg.norm(localx, 2)
    localy = np.cross(localz, localx)
    unity = localy/np.linalg.norm(localy, 2)
    T = np.column_stack((localx, unity, unitz, apex_pt))

    # N = cut_numpy.shape[0]
    # origin = cut_numpy[0,:]
    # localz = np.cross((cut_numpy[1,:] - origin), (cut_numpy[2,:] - origin))
    # unitz = localz / np.linalg.norm(localz,2)
    # localx = cut_numpy[1,:] - origin
    # unitx = localx / np.linalg.norm(localx,2)
    # localy = np.cross(localz, localx)
    # unity = localy/np.linalg.norm(localy,2)
    # T = np.column_stack((localx, unity, unitz, origin))

    T = np.vstack((T, [0, 0, 0, 1]))
    C = np.column_stack((cut_numpy, np.ones((N, 1))))
    coor2D, _, _, _ = np.linalg.lstsq(T, np.transpose(C))
    coor2D = np.transpose(coor2D[0:2, :])

    return coor2D

# project on to xy plane! (important one)

def project_3d_points_to_plane(points, p1, p2 ,p3, numpoints):
    """
    Projects the points in 'array' onto a 3d plane defined by the points
    p1, p2 and p3.

    Inputs:
        array : ndarray (n_pts x 3)
        p1, p2, p3: ndarray (3 x 1)
        numpoints : number of points
    Returns:
        projected : ndarray (3 x 1)
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    # get vectors in plane
    v1 = p3 - p1
    v2 = p2 - p1

    # compute cross product
    cp = np.cross(v1, v2)
    a, b, c = cp # normal to plane is ax + by + cz

    # evaluate d
    d = np.dot(cp, p3)

    # thus, normal is given by
    plane = vtk.vtkPlane()
    origin = p1
    normal = normalize(np.array([a,b,c]))
    plane.SetOrigin(p1)
    plane.SetNormal(normal)

    if numpoints == 1:
        proj = [0,0,0]
        plane.ProjectPoint(points, origin, normal, proj)
        return proj
    else:
        projected_pts = np.zeros((numpoints, 3), dtype=float)

        for i in range(numpoints):
            proj = [0,0,0]
            plane.ProjectPoint(points[i], origin, normal, proj)
            projected_pts[i] = proj

        return projected_pts

# doesnt matter since the pt will already be in the 2d plane

def rotate_by_y_axis(points2d):
    """
    Rotates the 2d points by aligning the old basis to [1,0] and [0,1]
    Run this after "project_onto_xy_plane()"
    """
    # define rotation matrix
    transf_matrix = np.column_stack(([1,0], [0,1]))

    return np.dot(transf_matrix, points2d.T).T

def project_onto_xy_plane(points3d):
    """
    Correct version:

        This function returns 3d points projected onto xy plane
        whilst maintaing distance and scale parameters between points.
        https://math.stackexchange.com/questions/375995/rotate-3d-plane

        INPUT:
            points3d : 3d points of the view
            note: the points are ordered from top_point_1 to top_point_2

        OUTPUT:
            points2d : points3d rotated and translated to xy plane
                        such that z component is 0
            R : the final transformation matrix used for the projection onto the 3d plane.
            post_n : normal of the final plane
    """
    points3d = np.asarray(points3d, dtype=np.float)
    num_pts = points3d.shape[0]

    # translate all points so that COM is at the origin
    com = np.mean(points3d, axis=0)
    c = -1.0*com # vector pointing towards origin
    pts_trans = points3d + np.tile(c, (num_pts, 1))

    # compute normal of plane defined by top_pt1, top_pt2 and low_pt
    crssAxB = np.cross(pts_trans[0], pts_trans[-1])
    n = crssAxB / np.linalg.norm(crssAxB)
    nx = n[0]
    ny = n[1]
    nz = n[2]

    # construct Rz ..
    nx2ny2 = np.sqrt(nx**2 + ny**2)
    Rz = np.array([[nx / nx2ny2, ny / nx2ny2, 0],
                [-ny / nx2ny2, nx / nx2ny2, 0],
                [0, 0, 1]])

    # construct Ry ..
    n2 = np.dot(Rz, n)
    Ry = np.array([[n2[2], 0, -n2[0]],
                [0, 1, 0],
                [n2[0], 0, n2[2]]])

    # apply Rz and Ry ..
    R = np.dot(Ry, Rz)
    coords_xy = np.dot(R, pts_trans.T).T

    # ignore z component (all zeros after projection to xy plane)
    coords_xy = coords_xy[:,0:2]

    return coords_xy

def display_feature_points(polydata, highest_points, lowest_point):
    """
    Create actor for the poly data 2ch
    """

    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(polydata)

    cutM = vtk.vtkPolyDataMapper()
    cutM.SetInputConnection(vertexFilter.GetOutputPort())

    planeActor = vtk.vtkActor()
    planeActor.SetMapper(cutM)
    planeActor.GetProperty().SetColor([1, 0, 0])
    planeActor.GetProperty().SetLineWidth(6)

    # include feature points
    highest_points_a = include_points(list(highest_points), 2, 7, (0, 1, 0))
    lowest_point_a = include_points(list(lowest_point), 1, 7, (0, 0, 1))

    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0)
    ren.AddActor(highest_points_a)
    ren.AddActor(lowest_point_a)
    ren.AddActor(planeActor)

    vtk_show(ren)

def display_left_right_pts(left_pts, right_pts, polydata):
    """
    Function to display the landmarks in vtk formnat.
    """

    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(polydata)

    cutM = vtk.vtkPolyDataMapper()
    cutM.SetInputConnection(vertexFilter.GetOutputPort())

    planeActor = vtk.vtkActor()
    planeActor.SetMapper(cutM)
    planeActor.GetProperty().SetColor([1, 0, 0])
    planeActor.GetProperty().SetLineWidth(6)

    leftA = include_points(list(left_pts), len(left_pts), 7, (1, 0, 1))
    rightA = include_points(list(right_pts), len(right_pts), 7, (1, 1, 0))

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0)
    ren.AddActor(leftA)
    ren.AddActor(rightA)
    ren.AddActor(planeActor)

    return vtk_show(ren)

def display_horizontal_lines(horiz_a, horiz_b, cutPoly, disp):
    """
                                    !! DEPRECATED !!

    ALREADY IMPLEMENTED IN get_horizontal_points_interp(..,...,display_opt)

    1) Display points
    """

    VTK_horiz_all = vtk.vtkPoints()
    numpoints = len(horiz_a)

    for i in range(numpoints):
        VTK_horiz_all.InsertNextPoint(horiz_a[i])
        VTK_horiz_all.InsertNextPoint(horiz_b[i])

    horizPoly = vtk.vtkPolyData()
    horizPoly.SetPoints(VTK_horiz_all)

    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(horizPoly)

    horizM = vtk.vtkPolyDataMapper()
    horizM.SetInputConnection(vertexFilter.GetOutputPort())

    horizA = vtk.vtkActor()
    horizA.SetMapper(horizM)
    horizA.GetProperty().SetColor(0.0, 0.0, 0.0)
    horizA.GetProperty().SetPointSize(5)

    """
    2) Display endocardial mesh.
    """
    numPoints = len(cutPoly)
    vtk_float_arr = numpy_support.numpy_to_vtk(num_array=np.asarray(cutPoly), deep=True, array_type=vtk.VTK_FLOAT)
    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(vtk_float_arr)

    # add the basal points that connect from top_right to top_left
    basal_pts = getEquidistantPoints(cutPoly[0], cutPoly[-1], 100, 0, 0)
    num_basal_pts = len(basal_pts)
    for basal_pt in basal_pts:
        vtkpts.InsertNextPoint(basal_pt)

    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(numPoints + num_basal_pts)

    for i in range(numPoints + num_basal_pts):
        polyLine.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyLine)

    # add points and lines to polydata container
    cut_endo_poly = vtk.vtkPolyData()
    cut_endo_poly.SetPoints(vtkpts)
    cut_endo_poly.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cut_endo_poly)

    planeActor = vtk.vtkActor()
    planeActor.SetMapper(mapper)
    planeActor.GetProperty().SetColor(0, 0, 0)
    planeActor.GetProperty().SetLineWidth(2)
    """
    Display lines.
    """

    lineArray = vtk.vtkCellArray()

    for i in range(numpoints):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i + i)
        line.GetPointIds().SetId(1, i+1 + i)
        lineArray.InsertNextCell(line)

    # create polydata
    polyLine = vtk.vtkPolyData()

    # add points and lines to polydata container
    polyLine.SetPoints(VTK_horiz_all)
    polyLine.SetLines(lineArray)

    # create pipeline for display*
    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputData(polyLine)

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetColor(0, 0, 1)
    lineActor.GetProperty().SetLineWidth(5)

    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AddActor(horizA)
    ren.AddActor(planeActor)
    ren.AddActor(lineActor)


    if disp == 1:
        vtk_show(ren)

def split_window(num_x, num_y):
    """
    num_x = number of boxes in the x direction

    (xmin, ymin, xmax, ymax) --> locations of left and right corner positions starting from top of square
    """

    viewPorts = np.zeros((num_x*num_y, 4))
    counter = 0
    for i in range(num_x):
        for j in range(num_y):
            viewPorts[num_x*j + i, :] = [i*(1.0/float(num_x)),
                                         1.0 - (j+1.0)*(1.0/float(num_y)),
                                         (i+1.0)*1.0/float(num_x),
                                         1.0 - j*(1.0/float(num_y))
                                         ]  # [num_y*i + j,:] for vertical fill first

    return viewPorts

def vtk_multiple_renderers(renderers, width, height):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """

    # create renderer window
    window = vtk.vtkRenderWindow()
    for i in range(len(renderers)):
        window.AddRenderer(renderers[i])

    window.SetSize(width, height)

    # create render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()

    # start the display
    window.Render()
    interactor.Start()

def display_heatmap(circle_str, array, title):

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(array, cmap='gray')

    ax.set_xticks(np.arange(len(circle_str)))
    ax.set_yticks(np.arange(len(circle_str)))
    ax.set_xticklabels(circle_str, rotation=50)
    ax.set_yticklabels(circle_str)
    fig.colorbar(im)

    ax.set_title(title)
    ax.set_xlabel('offsets 2ch')
    ax.set_ylabel('offsets 4ch')
    fig.tight_layout()
    plt.show()

def get_vtktransform_rotate_global_to_local(v1, v2, v3):
    """
    Builds rotation matrix to rotate from global coordinate system x,y,z to local coord. system

    Returns transform vtk object to apply directly to actors
    """


    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])

    rot_mat = np.zeros((3, 3), dtype=float)
    rot_mat[0, 0] = np.dot(v1, x)
    rot_mat[0, 1] = np.dot(v1, y)
    rot_mat[0, 2] = np.dot(v1, z)
    rot_mat[1, 0] = np.dot(v2, x)
    rot_mat[1, 1] = np.dot(v2, y)
    rot_mat[1, 2] = np.dot(v2, z)
    rot_mat[2, 0] = np.dot(v3, x)
    rot_mat[2, 1] = np.dot(v3, y)
    rot_mat[2, 2] = np.dot(v3, z)

    rot_mat = np.column_stack((rot_mat, np.array([0.0, 0.0, 0.0])))
    rot_mat = np.vstack((rot_mat, np.array([0.0, 0.0, 0.0, 1.0])))

    vtkM = vtk.vtkMatrix4x4()

    for i in range(4):
        for j in range(4):
            vtkM.SetElement(i, j, rot_mat[i, j])

    transform = vtk.vtkTransform()
    transform.PreMultiply()
    transform.SetMatrix(vtkM)

    return transform

def compute_convex_hull_area(polydata):

    coor2D = convert_3d_plane_to_2d(polydata)  # convert to 2D first
    hull = ConvexHull(coor2D)

    # plt.figure(figsize=(6,6))
    # plt.plot(coor2D[:,0], coor2D[:,1], 'o')
    # for simplex in hull.simplices:
    #     plt.plot(coor2D[simplex, 0], coor2D[simplex, 1], 'k-')
    #
    # plt.show(block=False)

    return hull.volume  # since this is 2D volume actually returns the area

def distance(P1, P2):
    """
    This function computes the distance between 2 points defined by P1 = (x1,y1) and P2 = (x2,y2)
    """
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

def optimized_path(coords, startid, mask):
    """
    This function finds the nearest point to a point coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...]
    """
    coords = np.column_stack((coords, mask))
    pass_by = np.asarray(coords)
    path = [coords[startid]]
    pass_by = np.delete(pass_by, startid, axis=0)
    while pass_by.any():
        nearest_id, nearest = min(
            enumerate(pass_by), key=lambda x: distance(path[-1][:2], x[1][:2]))
        path.append(nearest)
        pass_by = np.delete(pass_by, nearest_id, axis=0)

    return path
