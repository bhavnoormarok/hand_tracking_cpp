#include <Reg.h>


void Reg::calculate_phi( ){  //correct
    // shape blends
    auto temp = mano_S * beta;  
    Eigen::MatrixX3d v_s = v + temp.reshaped(3,temp.size()/3).transpose(); // (|v|, 3)
    // obtain keypoint positions in shaped mesh
    
    k_s.resize(K.rows(), v_s.cols());
    k_s = K * v_s;  // (21, 3)

    Eigen::VectorXd b_amano, b_data;
    calculate_bone_lengths(k_s, b_amano);
    calculate_bone_lengths(k_marked, b_data);

    phi.resize(b_data.size());
    phi = b_data.array() / b_amano.array();

    return;
}



void Reg::compute_global_trans_from_palm_keypoints(std::vector<Eigen::Vector3d> k_data_palm, std::vector<Eigen::Vector3d> k_model_palm){   //correct
    // use point-to-point registration 
    auto pcd_k_data_palm = open3d::geometry::PointCloud(k_data_palm);
    auto pcd_k_model_palm = open3d::geometry::PointCloud(k_model_palm);

    std::vector<Eigen::Vector2i> corres;

    
    for (int i=0;i<k_data_palm.size();i++){
        Eigen::Vector2i a(i,i);
        corres.push_back(a);
    }
    

    Eigen::Matrix4d T_glob = open3d::pipelines::registration::TransformationEstimationPointToPoint().ComputeTransformation(pcd_k_model_palm, pcd_k_data_palm, corres);
    R_glob_init = T_glob(Eigen::seq(0,2), Eigen::seq(0,2));
    theta_orig->t_glob = T_glob(Eigen::seq(0,2), 3);
}


void Reg::compute_dof_mask_per_vert(){
    // for each vertex, identify dofs that it influences


 
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>  m_bone_per_vert =  W_bone.array() > 0.1;   // (||v||, 20)
    m_dof_per_vert.resize(W_bone.rows(),26);
    //cout<<W_bone.rows()<<" "<<W_bone.cols()<<endl;
    m_dof_per_vert = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(W_bone.rows(), 26, false); //np.full((len(W_bone), 26), False)  // (||v||, 26)
    for (int i_f=0;i_f<5;i_f++){
        int wrist_id = 4*i_f, mcp_id = wrist_id+1, pip_id = mcp_id+1, dip_id = pip_id+1;
        int i_t_mcp1 = 4*i_f, i_t_mcp2 = i_t_mcp1+1, i_t_pip = i_t_mcp2+1, i_t_dip = i_t_pip+1;

        m_dof_per_vert(Eigen::placeholders::all,0) = m_dof_per_vert(Eigen::placeholders::all,0) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));
        m_dof_per_vert(Eigen::placeholders::all,1) = m_dof_per_vert(Eigen::placeholders::all,1) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));
        m_dof_per_vert(Eigen::placeholders::all,2) = m_dof_per_vert(Eigen::placeholders::all,2) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));
        m_dof_per_vert(Eigen::placeholders::all,3) = m_dof_per_vert(Eigen::placeholders::all,3) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));
        m_dof_per_vert(Eigen::placeholders::all,4) = m_dof_per_vert(Eigen::placeholders::all,4) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));
        m_dof_per_vert(Eigen::placeholders::all,5) = m_dof_per_vert(Eigen::placeholders::all,5) || (m_bone_per_vert(Eigen::placeholders::all, wrist_id) || m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id));

        
        m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp1) = m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id);
        // cout<<m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp1).rows()<<" "<<m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp1).cols()<<endl;
        // cout<<m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp1).rows()<<" "<<m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp1).cols()<<endl;
        
        m_dof_per_vert(Eigen::placeholders::all, 6+i_t_mcp2) = m_bone_per_vert(Eigen::placeholders::all, mcp_id) || m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id);
        m_dof_per_vert(Eigen::placeholders::all, 6+i_t_pip) = m_bone_per_vert(Eigen::placeholders::all, pip_id) || m_bone_per_vert(Eigen::placeholders::all, dip_id);
        m_dof_per_vert(Eigen::placeholders::all, 6+i_t_dip) = m_bone_per_vert(Eigen::placeholders::all, dip_id);
    }

    return ;

}


void Reg::compute_dof_mask_per_face( ){
    // for each face, identify dofs that it influences

    m_dof_per_face.resize(F.rows(),26);
    m_dof_per_face = m_dof_per_vert(F(Eigen::placeholders::all, 0),Eigen::placeholders::all) || m_dof_per_vert(F(Eigen::placeholders::all, 1),Eigen::placeholders::all) || m_dof_per_vert(F(Eigen::placeholders::all, 2),Eigen::placeholders::all);    // (|F|, 26)
    return ;

}





void Reg::compute_face_ids_per_part(){
    Eigen::VectorXi p_v(W_bone.rows());

    for (int i=0;i<W_bone.rows();i++){
        W_bone.row(i).maxCoeff(&p_v(i)); 
    }

    Eigen::VectorXi p_F = p_v(F.col(0));


    Eigen::VectorXd V;
    n_f_per_part.resize((W_bone.cols()));
    
    int temp = 0;
    for (int i_p=0;i_p<W_bone.cols();i_p++){
        Eigen::VectorXi I,J;
        igl::find(p_F.array() == i_p,I,J,V);
        i_f_per_part.push_back(I);
        temp += I.size();
        n_f_per_part(i_p) = I.size();
    }

    return;
}




void Reg::set_params(){
    xyz_crop_center.resize(1,3);
    xyz_crop_center<<0.0,0.0,0.7;
    d_near=600.0; d_far=1200.0;  fx=366.085;  fy=366.085;  cx=259.229;  cy=207.968;
    n_x = 200;
    img_height = 424;
    img_width = 512;

    left_shift = log2(img_height)+log2(img_width)+1;


    label2x.resize(img_width*img_height+1), label2y.resize(img_width*img_height+1);



}

void Reg::set_Amano_vars(){
    igl::readOBJ("./../data/mesh.obj",v,F);
    compute_vertex_normals(v,F,n);        

    Eigen::read_binary("./../data/axis_per_dof.dat",axes);
    Eigen::read_binary_sparse("./../data/K.dat",K);
    Eigen::read_binary("./../data/W_bone.dat",W_bone);
    Eigen::read_binary("./../data/W_endpoint.dat",W_endpoint);
    Eigen::read_binary("./../data/mano_S.dat",mano_S);
    Eigen::read_binary("./../data/mano_P.dat",mano_P);

    i_Rs_rel_art_mano.resize(15); 
    i_Rs_rel_art_mano<<3,4,5,6,7,8,12,13,14,9,10,11,0,1,2;

    beta.resize(10); //= Eigen::VectorXd::Zero(10,0.0);
    beta<<-0.17704946,  0.0090871 , -0.00929303 , 0.00271509,  0.01484054 , 0.00099747,
           0.01197357 , 0.00385809 ,-0.00292281, -0.00319425;

    Eigen::read_binary("./../data/k_marked.dat",k_marked);

    //calculate_phi(k_marked, beta, phi, k_s);
    calculate_phi();

    i_k_amano_palm.resize(6);
    i_k_amano_palm<< 0, 1, 5, 9, 13, 17;

    std::vector<Eigen::Vector3d> k_s__, k_marked__;
    for (int i=0;i<i_k_amano_palm.size();i++){
        k_s__.push_back(k_s.row(i_k_amano_palm(i)));
        k_marked__.push_back(k_marked.row(i_k_amano_palm(i)));
    }

    compute_global_trans_from_palm_keypoints(k_marked__, k_s__);

    theta_orig->theta_glob = Eigen::Vector3d::Zero(3);
    theta_orig->theta = Eigen::VectorXd::Zero(20);
    k_p_prev = NULL;

    deform_and_compute_linearized_info(vnkap_orig, theta_orig);

    

}




void Reg::set_RegistrationBase_var(){
    //theta_bounds_dir = "./output/hand_model/prior/bounds";
    Eigen::read_binary("./../data/theta_min.dat",theta_min);
    Eigen::read_binary("./../data/theta_max.dat",theta_max);
    Eigen::read_binary("./../data/mu.dat",mu);
    Eigen::read_binary("./../data/Pi.dat",Pi);
    Eigen::read_binary("./../data/Sigma.dat",Sigma);


    i_s_per_pairs.resize(257,2);
    r_per_sphere.resize(28);
    i_v_per_sphere.resize(28,4);

    r_per_sphere << 0.008, 0.009, 0.0095, 0.0105,           
    0.0055, 0.007, 0.0075, 0.0082, 0.009, 0.0095,          
    0.006, 0.007, 0.0075, 0.0082, 0.009, 0.0095, 0.011,
    0.0055, 0.007, 0.0072, 0.008, 0.0085, 0.009, 
    0.005, 0.006, 0.0065, 0.007, 0.0085;  

    i_v_per_sphere<<727, 763, 748, 734,
                    731, 756, 749, 733,
                    708, 754, 710, 713,
                    250, 267, 249, 28,

                    350, 314, 337, 323,
                    343, 316, 322, 336,
                    342, 295, 299, 297,
                    280, 56, 222, 155,
                    165, 133, 174, 189,
                    136, 139, 176, 170,

                    462, 426, 460, 433,
                    423, 455, 448, 432,
                    430, 454, 457, 431,
                    397, 405, 390, 398,
                    357, 364, 391, 372,
                    375, 366, 381, 367,
                    379, 399, 384, 380,

                    573, 537, 560, 544,
                    566, 534, 559, 543,
                    565, 541, 542, 523,
                    507, 476, 501, 508,
                    496, 498, 491, 495,
                    489, 509, 494, 490,

                    690, 654, 677, 664,
                    682, 658, 642, 669,
                    581, 633, 619, 629,
                    614, 616, 609, 613,
                    607, 627, 612, 608;

    i_s_per_pairs<<0,4,0,5,0,6,0,7,0,8,0,10,0,11,0,12,0,13,0,14,0,15,0,17,0,18,0,19,0,20,0,21,0,23,0,24,0,25,0,26,1,4,1,5,1,6,1,7,1,8,1,10,1,11,1,12,1,13,1,14,1,15,1,17,1,18,1,19,1,20,1,21,1,23,1,24,1,25,1,26,2,4,2,5,2,6,2,7,2,8,2,10,2,11,2,12,2,13,2,14,2,15,2,17,2,18,2,19,2,20,2,21,2,23,2,24,2,25,2,26,3,4,3,5,3,6,3,7,3,8,3,10,3,11,3,12,3,13,3,14,3,15,3,17,3,18,3,19,3,20,3,21,3,23,3,24,3,25,3,26,4,10,4,11,4,12,4,13,4,14,4,15,4,17,4,18,4,19,4,20,4,21,4,23,4,24,4,25,4,26,5,10,5,11,5,12,5,13,5,14,5,15,5,17,5,18,5,19,5,20,5,21,5,23,5,24,5,25,5,26,6,10,6,11,6,12,6,13,6,14,6,15,6,17,6,18,6,19,6,20,6,21,6,23,6,24,6,25,6,26,7,10,7,11,7,12,7,13,7,14,7,15,7,17,7,18,7,19,7,20,7,21,7,23,7,24,7,25,7,26,8,10,8,11,8,12,8,13,8,14,8,15,8,17,8,18,8,19,8,20,8,21,8,23,8,24,8,25,8,26,9,10,9,11,9,12,9,13,9,14,9,15,9,17,9,18,9,19,9,20,9,21,9,23,9,24,9,25,9,26,10,17,10,18,10,19,10,20,10,21,10,23,10,24,10,25,10,26,11,17,11,18,11,19,11,20,11,21,11,23,11,24,11,25,11,26,12,17,12,18,12,19,12,20,12,21,12,23,12,24,12,25,12,26,13,17,13,18,13,19,13,20,13,21,13,23,13,24,13,25,13,26,14,17,14,18,14,19,14,20,14,21,14,23,14,24,14,25,14,26,15,17,15,18,15,19,15,20,15,21,15,23,15,24,15,25,15,26,16,17,16,18,16,19,16,20,16,21,16,23,16,24,16,25,16,26,17,23,17,24,17,25,17,26,18,23,18,24,18,25,18,26,19,23,19,24,19,25,19,26,20,23,20,24,20,25,20,26,21,23,21,24,21,25,21,26,22,23,22,24,22,25,22,26;


    compute_dof_mask_per_vert();
    compute_dof_mask_per_face();

    compute_face_ids_per_part();


    Eigen::read_binary("./../data/vertex_ids_surr_keypoints.dat",I_v_surr_k);

    m_dof_per_k.resize(I_v_surr_k.rows(),m_dof_per_vert.cols());
    m_dof_per_k = m_dof_per_vert(I_v_surr_k.col(0), Eigen::placeholders::all) || m_dof_per_vert(I_v_surr_k.col(1), Eigen::placeholders::all) || m_dof_per_vert(I_v_surr_k.col(2), Eigen::placeholders::all) || m_dof_per_vert(I_v_surr_k.col(3), Eigen::placeholders::all);

    
    i_k_amano_palm.resize(6);
    i_k_amano_palm<< 0, 1, 5, 9, 13, 17;


    return;
}




void Reg::set_Registration_vars(){
    this->w_pos = 1;
    this->w_nor = 1e-5;
    this->n_s_approx = 212; // = 212   //  212 results in 200 points
    this->w_data_3d = 1;
    this->w_data_2d = 0; //changed 
    this->w_theta_bound = 1e-4;
    this->w_pca = 1e-2;  // w_4 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
    this->w_pca_mean = 1e-3;    // w_5 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf

    Eigen::MatrixXd M_inv =  1.0+(w_pca_mean*Sigma*Sigma/w_pca).array();   // (20, 20)\

    int dim = Sigma.rows(); 
    this->I_minus_Pi_M_PiT.resize(dim,dim);
    this->I_minus_Pi_M_PiT = Eigen::MatrixXd::Identity(dim,dim)- Pi * (M_inv.diagonal().cwiseInverse()).asDiagonal() * Pi.transpose();       // (20, 20)

    this->w_int=1e-1;
    this->w_k_reinit=1e1;
    this->w_vel=1e-1;
    this->w_damp_init=1e-4;

    this->n_iter = 20;

    i_dof_update.resize(26);  // = np.arange(6+20)
    i_dof_update = Eigen::VectorXi::LinSpaced(26,0,25);

    i_k_kinect_reg_k.resize(21);
    i_k_kinect_reg_k << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20;

    i_k_amano_reg_k.resize(21);
    i_k_amano_reg_k << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20;

    i_k_amano_reinit.resize(21);
    i_k_amano_reinit << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20;

    int n_s_bg_approx = 500;

    generate_barycenters_on_mesh(i_F_bg, b_bg, m_dof_per_bg); //to be check


    __k_marked = new Eigen::MatrixX3d;
    *__k_marked = k_marked(i_k_kinect_reg_k, Eigen::placeholders::all);


}
