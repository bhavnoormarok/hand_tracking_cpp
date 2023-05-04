#pragma once

#include "../igl/readOBJ.h"
#include "../igl/find.h"
#include "open3d/Open3D.h"
#include <opencv2/opencv.hpp> 
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include<omp.h>



// Defining the ESCAPE Key Code




namespace Eigen{
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix){
        std::ofstream out(filename, std::ios::out || std::ios::binary || std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in_(filename);
        typename Matrix::Index rows=0, cols=0;
        in_.read((char*) (&rows),sizeof(typename Matrix::Index));
        in_.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in_.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in_.close();
    }


    template <class SparseMatrix>
    inline void write_binary_sparse(const std::string& filename, const SparseMatrix& matrix) {
        assert(matrix.isCompressed() == true);
        std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
        if(out.is_open())
        {
            typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
            rows = matrix.rows()     ;
            cols = matrix.cols()     ;
            nnzs = matrix.nonZeros() ;
            outS = matrix.outerSize();
            innS = matrix.innerSize();

            out.write(reinterpret_cast<char*>(&rows), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&cols), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&nnzs), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&outS), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&innS), sizeof(typename SparseMatrix::Index));

            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
            out.write(reinterpret_cast<const char*>(matrix.valuePtr()),       sizeScalar * nnzs);
            out.write(reinterpret_cast<const char*>(matrix.outerIndexPtr()),  sizeIndexS  * outS);
            out.write(reinterpret_cast<const char*>(matrix.innerIndexPtr()),  sizeIndexS  * nnzs);

            out.close();
        }
        else {
            std::cout << "Can not write to file: " << filename << std::endl;
        }
    }

    template <class SparseMatrix>
    inline void read_binary_sparse(const std::string& filename, SparseMatrix& matrix) {
        std::ifstream in(filename, std::ios::binary | std::ios::in);
        if(in.is_open()) {
            typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
            typename SparseMatrix::Index sizeIndex  = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Index       ));
            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            //std::cout << sizeScalar << " " << sizeIndex << std::endl;
            in.read(reinterpret_cast<char*>(&rows ), sizeIndex);
            in.read(reinterpret_cast<char*>(&cols ), sizeIndex);
            in.read(reinterpret_cast<char*>(&nnz  ), sizeIndex);
            in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
            in.read(reinterpret_cast<char*>(&inSz ), sizeIndex);

            matrix.resize(rows, cols);
            matrix.makeCompressed();
            matrix.resizeNonZeros(nnz);

            in.read(reinterpret_cast<char*>(matrix.valuePtr())     , sizeScalar * nnz  );
            in.read(reinterpret_cast<char*>(matrix.outerIndexPtr()), sizeIndexS * outSz);
            in.read(reinterpret_cast<char*>(matrix.innerIndexPtr()), sizeIndexS * nnz );

            matrix.finalize();
            in.close();
        } // file is open
        else {
            std::cout << "Can not open binary sparse matrix file: " << filename << std::endl;
        }
    }

}


void calculate_bone_lengths(Eigen::MatrixX3d& k, Eigen::VectorXd& x);
Eigen::MatrixX3d mat_rotate(Eigen::MatrixXd &Q, Eigen::MatrixX3d &v);
Eigen::MatrixXd compose_mat(Eigen::MatrixXd &Q1, Eigen::MatrixXd &Q2);
void compose_mat(Eigen::Matrix3d &Q1, Eigen::MatrixXd &Q2, Eigen::MatrixXd& res);
void compose_mat(Eigen::MatrixXd &Q1, Eigen::MatrixXd &Q2, Eigen::MatrixXd &res);
void uvd2xyz(Eigen::MatrixX3d& U, Eigen::MatrixX3d& X, double& fx, double& fy, double& cx, double& cy);
void xyz2uv(Eigen::MatrixX3d& X, Eigen::MatrixX2d& U, double& fx, double& fy, double& cx, double& cy);
void compute_J_skel_(Eigen::MatrixX3d& y, Eigen::MatrixXd& m_dof_per_y, Eigen::MatrixX3d& axis_per_dof, Eigen::MatrixX3d& pivot_per_dof, Eigen::MatrixXd& J_skel);
void compute_J_persp(Eigen::MatrixX3d& x_bg, double fx, double fy, Eigen::MatrixXd& J_persp);
void furthest_point_downsample_ids(std::vector<Eigen::Vector3d>& D_, int n_S, std::vector<size_t>& S_ids, Eigen::Matrix3Xd& D);
void compute_vertex_normals(Eigen::MatrixX3d& v ,Eigen::MatrixX3i& F, Eigen::MatrixX3d& n);
void sample_face_ids(std::vector<Eigen::VectorXi>& i_f_per_part, Eigen::VectorXi& n_f_per_part, int n_f, int n_s_approx, Eigen::VectorXi& i_F_s);
void generate_random_barycentric_coordinates(int n_s, Eigen::MatrixX3d& b);
void barycenters_to_mesh_positions(Eigen::MatrixX3d& b, Eigen::VectorXi& i_Fb, Eigen::MatrixX3d& v, Eigen::MatrixX3i& F, Eigen::MatrixX3d& pb);
void barycenters_to_mesh_normals(Eigen::MatrixX3d& b, Eigen::VectorXi& i_Fb, Eigen::MatrixX3d& n, Eigen::MatrixX3i& F, Eigen::MatrixX3d& nb);
// void compute_vertex_normals(Eigen::MatrixX3d& v ,Eigen::MatrixX3i& F, Eigen::MatrixX3d& n);
// void compose_mat(Eigen::Matrix3d &Q1, Eigen::MatrixXd &Q2, Eigen::MatrixXd&  res );
// Eigen::MatrixX3d mat_rotate(Eigen::MatrixXd &Q, Eigen::MatrixX3d &v);
// Eigen::MatrixXd compose_mat(Eigen::MatrixXd &Q1, Eigen::MatrixXd &Q2);
// void compose_mat(Eigen::MatrixXd &Q1, Eigen::MatrixXd &Q2, Eigen::MatrixXd &res);

struct vnkap{
    Eigen::MatrixX3d v_p, n_p, k_p;
    //Eigen::MatrixX3d * k_p_prev;
    Eigen::MatrixX3d axis_per_dof;
    Eigen::MatrixX3d pivot_per_dof;
};

struct theta{
    Eigen::Vector3d theta_glob; 
    Eigen::Vector3d t_glob; 
    Eigen::VectorXd theta;
};

struct y_data{
    Eigen::MatrixX3d y;
    Eigen::MatrixX3d yn;
    Eigen::VectorXi i_F_y; 
    Eigen::MatrixX3d b_y; 
    Eigen::MatrixXd m_dof_per_y;
};

struct xpq {
    Eigen::MatrixX3d  x_bg; 
    Eigen::MatrixX2d  p; 
    Eigen::MatrixX2d  q;
};

class Reg{

public:

    // parameters
    Eigen::MatrixX3d xyz_crop_center;     //declare static later
    double d_near, d_far,  fx,  fy,  cx,  cy;
    int n_x;

    int img_height;
    int img_width;

    int left_shift;


    Eigen::VectorXi label2x, label2y;

    // amano variables
    Eigen::MatrixX3d v;
    Eigen::MatrixX3i F;
    Eigen::MatrixX3d n;
    Eigen::MatrixX3d axes;
    Eigen::MatrixXd W_bone;
    Eigen::MatrixXd W_endpoint;
    Eigen::SparseMatrix<double> K;
    Eigen::MatrixX3d k;

    Eigen::VectorXi i_Rs_rel_art_mano;

    Eigen::MatrixXd mano_S;
    Eigen::MatrixXd mano_P;

    
    // curr frame variables

    Eigen::MatrixXd label0,label1;
    Eigen::Matrix3Xd x,xn,x_dense;

    y_data * y_data_orig; 
    y_data * y_data_new; 

    xpq * xpq_orig;
    xpq * xpq_new;

    Eigen::VectorXd phi; 
    Eigen::MatrixX3d k_s;
    Eigen::MatrixX3d k_marked;
    Eigen::VectorXd beta;

    Eigen::Matrix3d R_glob_init; 

    theta* theta_orig;
    theta* theta_new;

    // Eigen::Vector3d theta_glob; 
    // Eigen::Vector3d t_glob; 
    // Eigen::VectorXd theta;

    // Eigen::Vector3d theta_glob_new; 
    // Eigen::Vector3d t_glob_new; 
    // Eigen::VectorXd theta_new;

    // Eigen::MatrixX3d v_p, n_p, k_p;
    Eigen::MatrixX3d * k_p_prev;
    // Eigen::MatrixX3d axis_per_dof;
    // Eigen::MatrixX3d pivot_per_dof;

    vnkap* vnkap_orig;
    vnkap* vnkap_new;


    // Eigen::MatrixX3d v_p_new, n_p_new, k_p_new;
    // Eigen::MatrixX3d axis_per_dof_new;
    // Eigen::MatrixX3d pivot_per_dof_new;

    // Registration Base variables
    Eigen::VectorXd theta_min; //np.load(f'{theta_bounds_dir}/theta_min.npy')
    Eigen::VectorXd theta_max;
    Eigen::VectorXd mu;
    Eigen::MatrixXd Pi;
    Eigen::MatrixXd Sigma;

    Eigen::MatrixX2i i_s_per_pairs;
    Eigen::VectorXd r_per_sphere;
    Eigen::MatrixXi i_v_per_sphere;


    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_dof_per_vert;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_dof_per_face;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_dof_per_k;

    std::vector<Eigen::VectorXi> i_f_per_part;
    Eigen::VectorXi n_f_per_part;

    Eigen::MatrixXi I_v_surr_k;

    Eigen::VectorXi i_k_amano_palm;

    


    //hyperparameters
    double w_pos;
    double w_nor;
    int n_s_approx; // = 212   //  212 results in 200 points
    double w_data_3d; //  = 1
    double w_data_2d; 
    double w_theta_bound;
    double w_pca;  // w_4 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
    double w_pca_mean;    // w_5 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
    Eigen::MatrixXd I_minus_Pi_M_PiT;// = compute_I_minus_Pi_M_PiT(self.Pi, self.Sigma, self.w_pca, self.w_pca_mean)
    double w_int;
    double w_k_reinit;
    double w_vel;
    double w_damp_init;

    Eigen::VectorXi i_dof_update;  // = np.arange(6+20)

    Eigen::VectorXi i_k_kinect_reg_k, i_k_amano_reg_k, i_k_amano_reinit;

    int n_iter;

    Eigen::VectorXi whole_num;

    Eigen::VectorXi i_F_bg;
    Eigen::MatrixX3d b_bg; 
    Eigen::MatrixXd m_dof_per_bg;

    open3d::geometry::TriangleMesh  T;


    Eigen::MatrixX3d* __k_marked;

    Reg();
    ~Reg();
    void calculate_phi();
    void compute_global_trans_from_palm_keypoints(std::vector<Eigen::Vector3d> k_data_palm, std::vector<Eigen::Vector3d> k_model_palm);
    void compute_dof_mask_per_face();
    void compute_dof_mask_per_vert();
    void compute_face_ids_per_part();
    void generate_barycenters_on_mesh(Eigen::VectorXi& i_F_s,Eigen::MatrixX3d& b_s, Eigen::MatrixXd& m_dof_per_s);
    void set_params();
    void set_Amano_vars();
    void set_RegistrationBase_var();
    void set_Registration_vars();

    void process_frame_(cv::Mat& color_raw, cv::Mat& depth_raw, cv::Mat& depth_proc2);
    void compute_sil_idx_at_each_pixel(cv::Mat& depth_proc);
    void depth_to_point_cloud(cv::Mat& depth_proc);
    void load_frame(int i);
    void deform(theta* theta_,Eigen::MatrixX3d& v_art, Eigen::MatrixXd& Rs_rel_art, Eigen::MatrixXd& Rs,  Eigen::MatrixX3d& a_prime);
    void compute_axis_per_dof_in_world_space( Eigen::Matrix3d& R_glob_ref_x, Eigen::Matrix3d& R_glob_ref_y, Eigen::Matrix3d& R_glob_ref_z, Eigen::MatrixXd& Rs_rel, vnkap* vnkap_);
    void compute_pivot_per_dof_in_world_space(Eigen::Matrix3d& R_glob, Eigen::Vector3d& t_glob, Eigen::MatrixX3d& a_prime, vnkap* vnkap_);
    void deform_and_compute_linearized_info(vnkap * vnkap_, theta * theta_);
    void start_engine();

    void compute_theta_bound_terms(Eigen::VectorXd& theta, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    void compute_pca_prior_terms(Eigen::VectorXd& theta, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    void compute_intersection_penalty_term(vnkap * vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    void compute_velocity_terms(Eigen::MatrixX3d* k_p_prev, vnkap* vnkap_, Eigen::MatrixXd& m_dof_per_k, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    void compute_keypoint_terms(Eigen::MatrixX3d* k_data, Eigen::MatrixX3d& k_model, Eigen::MatrixXd& m_dof_per_k, vnkap* vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    
    void compute_energy_for_keypoints( vnkap * vnkap_, theta* theta_, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& E);

    void solve_(Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& w_damp, Eigen::VectorXd& dpose);
    void register_to_keypoints();


    void register_to_pointcloud();
    void compute_energy_for_pointcloud( vnkap * vnkap_, theta* theta_, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& E);

    void compute_2d_data_terms(xpq * xpq_, vnkap* vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);
    void update_3d_correspondences(vnkap* vnkap_, y_data* ydata_old, y_data* ydata_new);
    void compute_2d_correspondences(vnkap* vnkap_, xpq * xpq_);
    void compute_3d_correspondences(vnkap* vnkap_, y_data* ydata_);
    void compute_closest_point_on_mesh( Eigen::MatrixX3d& p_s, Eigen::MatrixX3d& n_s, Eigen::VectorXd& i_p_s_per_x, Eigen::VectorXd& d_x_p);
    void compute_3d_data_terms( y_data* ydata_, vnkap*vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte);

};
