#include<Reg.h>
#include <chrono>
using namespace std::chrono;


void Reg::compute_2d_data_terms(xpq * xpq_, vnkap* vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    // E = n.T * (J_persp @ J_skel @ dp + (p-q))
    // J = n.T * J_persp @ J_skel
    // e = n.T * (q-p)

    Eigen::MatrixXd J_skel;
    Eigen::MatrixXd J_persp;
    compute_J_skel_(xpq_->x_bg, m_dof_per_bg, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J_skel);    // (|x|, 3, 26)
    compute_J_persp(xpq_->x_bg, fx, fy, J_persp);             // (|x|, 2, 3)
    int aaa = xpq_->x_bg.rows();
    Eigen::MatrixXd J(2*aaa,26);

    // #pragma omp parallel for
    for (int i=0;i<aaa;i++){
        Eigen::MatrixXd temp = J_persp({i,i+aaa},Eigen::placeholders::all)*J_skel({i,i+aaa,i+2*aaa},Eigen::placeholders::all);
        //std::cout<<temp.rows()<<" "<<temp.cols()<<std::endl;
        J({i,i+aaa}, Eigen::placeholders::all) = temp;
    }
     
    e = (xpq_->q - xpq_->p).reshaped().cast<double>();    // (2*|x|,)
    JtJ = J.transpose() * J;   // (26, 26)
    Jte = J.transpose() * e;
    return;
}


//correct
void Reg::compute_3d_data_terms( y_data* ydata_, vnkap*vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    // E = n.T * (J_skel @ dp + (y-x))
    // J = n.T * J_skel
    // e = n.T * (x-y)
    Eigen::MatrixXd J_skel;
    int n = ydata_->y.rows();
    compute_J_skel_(ydata_->y, ydata_->m_dof_per_y, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J_skel);    // (|y|, 3, 26)
    

    Eigen::MatrixXd J_ = J_skel.array().colwise()*ydata_->yn.reshaped().array();
    Eigen::MatrixXd J = J_(Eigen::seq(0,n-1), Eigen::placeholders::all)+J_(Eigen::seq(n,2*n-1), Eigen::placeholders::all)+J_(Eigen::seq(2*n,3*n-1), Eigen::placeholders::all);
    //std::cout<<J<<"oiu"<<std::endl;


    e = (ydata_->yn.array() * (x.transpose() - ydata_->y).array()).rowwise().sum();    // (|y|,)
    JtJ = J.transpose() * J;   // (26, 26)
    Jte = J.transpose() * e;   // (26,)

    return;
}



void Reg::compute_theta_bound_terms(Eigen::VectorXd& theta, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    // E = mask_max (dtheta + theta - theta_max) + mask_min (dtheta + theta - theta_min)
    // J = mask_max + min_mask
    // e = mask_max (theta_max - theta) + mask_min (theta_min - theta)
    

    Eigen::VectorXd theta_thetamax = theta-theta_max;

    
    Eigen::VectorXd theta_thetamin = theta-theta_min;
    auto mask_max = theta_thetamax.array() > 0;
    auto mask_min = theta_thetamin.array() < 0;
    e = (mask_max).select(theta_thetamax,0) + (mask_min).select(theta_thetamin,0); //(20,)

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(20, 26);

    auto temp = (-1.0*mask_max.cast<double>() - mask_min.cast<double>()).matrix();
    //auto temp  = Eigen::VectorXd::Constant(20,0.0);
    
    J(Eigen::placeholders::all, Eigen::seq(6,25)).diagonal() = temp;  // (20, 20)

    Eigen::MatrixXd JT = J.transpose();

    JtJ.resize(26,26);
    JtJ = JT * J;   // (26, 26)

    Jte.resize(26);
    Jte = JT * e;   // (26,)

    return;
}

//correct
void Reg::compute_pca_prior_terms(Eigen::VectorXd& theta, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    // E = (I - Pi @ M @ Pi.T)(dtheta + theta - mu)
    // J = (I - Pi @ M @ Pi.T)
    // e = (I - Pi @ M @ Pi.T) @ (mu - theta)

    e.resize(I_minus_Pi_M_PiT.rows());
    e = I_minus_Pi_M_PiT * (mu - theta);  // (20,)
    
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(e.size(), 26);  // (20, 26)
    J(Eigen::placeholders::all, Eigen::seq(6,25)) = I_minus_Pi_M_PiT;

    JtJ.resize(26,26);
    Jte.resize(26);

    Eigen::MatrixXd JT = J.transpose();

    JtJ = JT * J;   // (26, 26)
    Jte = JT * e;   // (26,)



    return;
}






// correct
void Reg::compute_intersection_penalty_term(vnkap * vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    Eigen::MatrixXd a1,a2,a3,a4;
    a1 = vnkap_->v_p(i_v_per_sphere(Eigen::placeholders::all,0), Eigen::placeholders::all);
    a2 = vnkap_->v_p(i_v_per_sphere(Eigen::placeholders::all,1), Eigen::placeholders::all);
    a3 = vnkap_->v_p(i_v_per_sphere(Eigen::placeholders::all,2), Eigen::placeholders::all);
    a4 = vnkap_->v_p(i_v_per_sphere(Eigen::placeholders::all,3), Eigen::placeholders::all);
    auto c_per_sphere = (a1+a2+a3+a4)/4;

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> b1,b2,b3,b4;
    b1 = m_dof_per_vert(i_v_per_sphere(Eigen::placeholders::all,0), Eigen::placeholders::all);
    b2 = m_dof_per_vert(i_v_per_sphere(Eigen::placeholders::all,1), Eigen::placeholders::all);
    b3 = m_dof_per_vert(i_v_per_sphere(Eigen::placeholders::all,2), Eigen::placeholders::all);
    b4 = m_dof_per_vert(i_v_per_sphere(Eigen::placeholders::all,3), Eigen::placeholders::all);
    auto m_dof_per_sphere = b1.cwiseMax(b2).cwiseMax(b3).cwiseMax(b4);

    auto c1 = c_per_sphere(i_s_per_pairs.col(0),Eigen::placeholders::all);   // (257, 3)
    auto c2 = c_per_sphere(i_s_per_pairs.col(1),Eigen::placeholders::all);   // (257, 3)

    // radii for each pair
    Eigen::VectorXd r1 = r_per_sphere(i_s_per_pairs.col(0));   // (257,)
    Eigen::VectorXd r2 = r_per_sphere(i_s_per_pairs.col(1));   // (257,)
    Eigen::Matrix<bool,-1,-1> m_dof_per_c1 = m_dof_per_sphere(i_s_per_pairs.col(0),Eigen::placeholders::all);// (257, 26)
    Eigen::Matrix<bool,-1,-1> m_dof_per_c2 = m_dof_per_sphere(i_s_per_pairs.col(1),Eigen::placeholders::all); // (257, 26)
    Eigen::MatrixXd n1 = (c2 - c1).rowwise().normalized();    // (257, 3) 
    Eigen::MatrixXd n2 = -n1;   // (257, 3)
    //n1.array().colwise()*r1.array();
    Eigen::MatrixX3d x1 = c1.array() + n1.array().colwise()*r1.array();     // (257, 3)
    Eigen::MatrixX3d x2 = c2.array() + n2.array().colwise()*r2.array();     // (257, 3)

    Eigen::MatrixXd J_skel_x1, J_skel_x2;
    Eigen::MatrixXd m_dof_per_c1_ = m_dof_per_c1.cast <double> ();
    Eigen::MatrixXd m_dof_per_c2_ = m_dof_per_c2.cast <double> ();
    compute_J_skel_(x1, m_dof_per_c1_, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J_skel_x1);    // (257, 3, 26)
    compute_J_skel_(x2, m_dof_per_c2_, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J_skel_x2);    // (257, 3, 26)

    auto J_diff = (J_skel_x1 - J_skel_x2).array().colwise() * n1.array().reshaped();
    int len = J_diff.rows()/3;
    Eigen::MatrixXd J = J_diff(Eigen::seq(0,len-1),Eigen::placeholders::all)+J_diff(Eigen::seq(len, 2*len-1),Eigen::placeholders::all)+J_diff(Eigen::seq(2*len, 3*len-1),Eigen::placeholders::all);
    e = n1.cwiseProduct(x2 - x1).rowwise().sum();

    auto intersection_amount = (r1 + r2).array().square() - (c1 - c2).array().square().rowwise().sum();   // (257,)
    auto m_int = intersection_amount <= 0; // (257,)

    Eigen::VectorXd zer =  Eigen::VectorXd::Zero(J.cols());

    // #pragma omp parallel for
    for (int ii=0;ii<m_int.size();ii++){
        if (m_int(ii)){
            J.row(ii) = zer;
        }
    }

    e = (m_int).select(0,e);


    JtJ = J.transpose()*J;   // (26, 26)


    Jte = J.transpose() * e ;  // (26,)

    return;
}

//correct
void Reg::compute_velocity_terms(Eigen::MatrixX3d* k_p_prev, vnkap* vnkap_, Eigen::MatrixXd& m_dof_per_k, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    Eigen::MatrixXd  J;
    compute_J_skel_(vnkap_->k_p, m_dof_per_k, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J);  // (21*3, 26)

    e = (*k_p_prev - vnkap_->k_p).reshaped();  // (3*21,)

    JtJ = J.transpose() * J;   // (26, 26)


    Jte = J.transpose() * e;   // (26,)

 

    return;
}

void Reg::compute_keypoint_terms(Eigen::MatrixX3d* k_data, Eigen::MatrixX3d& k_model, Eigen::MatrixXd& m_dof_per_k, vnkap* vnkap_, Eigen::VectorXd& e, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte){
    // E = J_skel(k_model) @ dpose + (k_model - k_data)
    // J = J_skel(k_model)
    // e = k_data - k_model
    Eigen::MatrixXd J;
    compute_J_skel_(k_model, m_dof_per_k, vnkap_->axis_per_dof, vnkap_->pivot_per_dof, J);  // ( 3, 10, 26)
    e = (*k_data - k_model).reshaped();  // (3*10,)
    JtJ = J.transpose() * J;   // (26, 26)
    Jte = J.transpose() * e;   // (26,)

    return;
}


// correct
void Reg::compute_energy_for_keypoints(vnkap * vnkap_, theta* theta_, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& E){
    Eigen::VectorXd e_theta_bound; Eigen::MatrixXd JtJ_theta_bound; Eigen::VectorXd Jte_theta_bound;
    Eigen::VectorXd e_pca; Eigen::MatrixXd JtJ_pca; Eigen::VectorXd Jte_pca;
    Eigen::VectorXd e_int; Eigen::MatrixXd JtJ_int; Eigen::VectorXd Jte_int;
    Eigen::VectorXd e_k_reinit; Eigen::MatrixXd JtJ_k_reinit; Eigen::VectorXd Jte_k_reinit;

    compute_theta_bound_terms(theta_->theta, e_theta_bound, JtJ_theta_bound, Jte_theta_bound);
    compute_pca_prior_terms(theta_->theta, e_pca, JtJ_pca, Jte_pca);         //correct
    compute_intersection_penalty_term(vnkap_, e_int, JtJ_int, Jte_int);
    
    Eigen::MatrixX3d k_p_ = vnkap_->k_p(i_k_amano_reg_k, Eigen::placeholders::all);
    Eigen::MatrixXd m_dof_per_k_ = m_dof_per_k.cast <double> ();
    Eigen::MatrixXd m_dof_per_k__ = m_dof_per_k_(i_k_amano_reg_k, Eigen::placeholders::all);

    compute_keypoint_terms(__k_marked, k_p_, m_dof_per_k__, vnkap_, e_k_reinit, JtJ_k_reinit, Jte_k_reinit);

  

    JtJ = w_theta_bound*JtJ_theta_bound + w_pca*JtJ_pca + w_int*JtJ_int + w_k_reinit*JtJ_k_reinit;
    Jte = w_theta_bound*Jte_theta_bound + w_pca*Jte_pca + w_int*Jte_int + w_k_reinit*Jte_k_reinit;
    E = w_theta_bound*e_theta_bound.array().square().sum() + w_pca*e_pca.array().square().sum() + w_int*e_int.array().square().sum() + w_k_reinit*e_k_reinit.array().square().sum(); 

}


//correct except velocity and 2d
void Reg::compute_energy_for_pointcloud( vnkap * vnkap_, theta* theta_, Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& E){
    Eigen::VectorXd e_theta_bound; Eigen::MatrixXd JtJ_theta_bound; Eigen::VectorXd Jte_theta_bound;
    Eigen::VectorXd e_pca; Eigen::MatrixXd JtJ_pca; Eigen::VectorXd Jte_pca;
    Eigen::VectorXd e_int; Eigen::MatrixXd JtJ_int; Eigen::VectorXd Jte_int;
    Eigen::VectorXd e_k_reinit; Eigen::MatrixXd JtJ_k_reinit; Eigen::VectorXd Jte_k_reinit;
    Eigen::VectorXd e_data_3d; Eigen::MatrixXd JtJ_data_3d; Eigen::VectorXd Jte_data_3d;
    Eigen::VectorXd e_data_2d; Eigen::MatrixXd JtJ_data_2d; Eigen::VectorXd Jte_data_2d;
    Eigen::VectorXd e_vel; Eigen::MatrixXd JtJ_vel; Eigen::VectorXd Jte_vel;

    compute_3d_data_terms(y_data_orig, vnkap_, e_data_3d, JtJ_data_3d, Jte_data_3d);

    compute_2d_data_terms(xpq_orig, vnkap_, e_data_2d, JtJ_data_2d, Jte_data_2d);

    compute_theta_bound_terms(theta_->theta, e_theta_bound, JtJ_theta_bound, Jte_theta_bound);

    compute_pca_prior_terms(theta_->theta, e_pca, JtJ_pca, Jte_pca);

    compute_intersection_penalty_term(vnkap_, e_int, JtJ_int, Jte_int);
  

    JtJ = w_data_3d*JtJ_data_3d + w_data_2d*JtJ_data_2d + w_theta_bound*JtJ_theta_bound + w_pca*JtJ_pca + w_int*JtJ_int;
    Jte = w_data_3d*Jte_data_3d + w_data_2d*Jte_data_2d + w_theta_bound*Jte_theta_bound + w_pca*Jte_pca + w_int*Jte_int;
    E = w_data_3d*e_data_3d.array().square().sum() + w_data_2d*e_data_2d.array().square().sum() + w_theta_bound*e_theta_bound.array().square().sum() + w_pca*e_pca.array().square().sum() + w_int*e_int.array().square().sum();
    
    if(__k_marked != NULL){
        Eigen::MatrixX3d k_p_ = vnkap_->k_p(i_k_amano_reinit, Eigen::placeholders::all);
        Eigen::MatrixXd m_dof_per_k_ = m_dof_per_k.cast <double> ();
        Eigen::MatrixXd m_dof_per_k__ = m_dof_per_k_(i_k_amano_reinit, Eigen::placeholders::all);

        Eigen::VectorXd e_k_reinit; Eigen::MatrixXd JtJ_k_reinit; Eigen::VectorXd Jte_k_reinit;
        compute_keypoint_terms(__k_marked, k_p_, m_dof_per_k__, vnkap_, e_k_reinit, JtJ_k_reinit, Jte_k_reinit);
        JtJ += w_k_reinit*JtJ_k_reinit;
        Jte += w_k_reinit*Jte_k_reinit;
        E += w_k_reinit*e_k_reinit.array().square().sum();
    }

    if (k_p_prev == NULL){
        ;
    }
    else{
        Eigen::MatrixXd m_dof_per_k_ = m_dof_per_k.cast <double> ();
        compute_velocity_terms(k_p_prev, vnkap_, m_dof_per_k_, e_vel, JtJ_vel, Jte_vel);
        JtJ += w_vel*JtJ_vel;
        Jte += w_vel*Jte_vel;

        E += w_vel*e_vel.array().square().sum();
    }


}
