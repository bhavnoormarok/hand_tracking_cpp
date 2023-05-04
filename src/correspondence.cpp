#include <Reg.h>




void Reg::compute_closest_point_on_mesh( Eigen::MatrixX3d& p_s, Eigen::MatrixX3d& n_s, Eigen::VectorXd& i_p_s_per_x, Eigen::VectorXd& d_x_p){
    // for each data point, compute distances wrt all model points
    //check if this can be done with complex eigen matrices 

    //std::cout<<x.rows()<<" "<<x.cols()<<" "<<p_s.cols()<<" "<<p_s.rows()<<std::endl;
    Eigen::ArrayXXd D_x_p_pos(x.cols(),p_s.rows()), D_x_p_nor(x.cols(),p_s.rows());

    // #pragma omp parallel for
    for (int i=0;i<x.cols();i++){
        // #pragma omp parallel for
        for(int j=0;j<p_s.rows();j++){
            // std::cout<<(x.col(i).transpose()-p_s.row(j)).array().sum()<<std::endl;;
            D_x_p_pos(i,j) = (x.col(i).transpose()-p_s.row(j)).array().square().sum();
            D_x_p_nor(i,j) = (xn.col(i).transpose()-n_s.row(j)).array().square().sum();
        }
    }
   
    Eigen::ArrayXXd D_x_p = w_pos * D_x_p_pos + w_nor * D_x_p_nor;    // (|x|, |p_s|)

    // //corresponding model point is the point with minimum distance

    // std::cout<<D_x_p<<std::endl;
    i_p_s_per_x.resize(D_x_p.rows());
    d_x_p.resize(D_x_p.rows());

    // #pragma omp parallel for
    for (int i=0; i<D_x_p.rows();i++){
        d_x_p(i) = D_x_p.row(i).minCoeff(&i_p_s_per_x(i));  // (|x|,)
    }
}



//correct
void Reg::compute_3d_correspondences(vnkap* vnkap_, y_data* ydata_){
    // sample barycenters
    Eigen::VectorXi i_F_s;
    Eigen::MatrixX3d b_s;
    Eigen::MatrixXd m_dof_per_s;

    generate_barycenters_on_mesh( i_F_s, b_s, m_dof_per_s);

    // evaluate barycenters on given mesh
    Eigen::MatrixX3d p_s;
    barycenters_to_mesh_positions(b_s, i_F_s, vnkap_->v_p, F, p_s);

    Eigen::MatrixX3d n_s;
    barycenters_to_mesh_normals(b_s, i_F_s, vnkap_->n_p, F, n_s);   //n_s different because of vnkap_->n_p




    Eigen::VectorXd i_p_s_per_x;
    Eigen::VectorXd d_x_p;
    compute_closest_point_on_mesh( p_s, n_s, i_p_s_per_x, d_x_p);

    // std::cout<<p_s<<std::endl<<std::endl;
    // std::cout<<i_p_s_per_x<<std::endl<<std::endl;
    // std::cout<<d_x_p<<std::endl<<std::endl;
    // std::cout<<p_s(i_p_s_per_x, Eigen::placeholders::all)<<std::endl;
    ydata_->y = p_s(i_p_s_per_x, Eigen::placeholders::all);    // (|x|, 3)
    ydata_->yn = n_s(i_p_s_per_x, Eigen::placeholders::all);  // (|x|, 3)
    ydata_->i_F_y = i_F_s(i_p_s_per_x);  // (|x|,)
    ydata_->b_y = b_s(i_p_s_per_x, Eigen::placeholders::all);  // (|x|, 3)
    ydata_->m_dof_per_y = m_dof_per_s(i_p_s_per_x, Eigen::placeholders::all);  // (|x|, 26)

}


void Reg::compute_2d_correspondences(vnkap* vnkap_, xpq * xpq_){
    // evaluate barycenters on given mesh

    barycenters_to_mesh_positions(b_bg, i_F_bg, vnkap_->v_p, F, xpq_->x_bg);  // (|b_bg|, 3)
  
    // std::cout<<b_bg<<std::endl;
    // std::cout<<"rg"<<std::endl;
    // exit(0);




    xyz2uv(xpq_->x_bg, xpq_->p, fx, fy, cx, cy); // (|b_bg|, 2)



    // // // compute closest point on silhouette for each p
    xpq_->p = xpq_->p.array().max(0);
    xpq_->p.col(0) = xpq_->p.col(0).array().min(label0.cols()-1);
    xpq_->p.col(1) = xpq_->p.col(1).array().min(label0.rows()-1);
    // // // p[:, 0] = np.clip(p[:, 0], 0, I_D_vu.shape[1]-1); p[:, 1] = np.clip(p[:, 1], 0, I_D_vu.shape[0]-1)  // avoid point to project outside image bounds
    
    xpq_->q.resize(xpq_->x_bg.rows(),2);
    // xpq_->q.col(1) = label0(xpq_->p.col(1), xpq_->p.col(0)); // (|b_bg|, 2)
    // xpq_->q.col(0) = label1(xpq_->p.col(1), xpq_->p.col(0)); 
    
    Eigen::MatrixX2i p = xpq_->p.cast<int>();

    // #pragma omp parallel for
    for (int i=0;i<xpq_->q.rows();i++){
        xpq_->q(i,1) = label0(p(i,1),p(i,0));
        xpq_->q(i,0) = label1(p(i,1),p(i,0));
    }
 
    return;
}


void Reg::update_3d_correspondences(vnkap* vnkap_, y_data* ydata_old, y_data* ydata_new){
    // evaluate position for previous barycenters for new pose
    Eigen::MatrixX3d y, yn;
    // std::cout<<"ytrewq32"<<std::endl;
    barycenters_to_mesh_positions(ydata_old->b_y, ydata_old->i_F_y, vnkap_->v_p, F, y);


    barycenters_to_mesh_normals(ydata_old->b_y, ydata_old->i_F_y, vnkap_->n_p, F, yn);


    // std::cout<<x.rows()<<" "<<y.rows()<<" "<<x.cols()<<" "<<y.cols()<<std::endl;
    // std::cout<<xn.rows()<<" "<<yn.rows()<<" "<<xn.cols()<<" "<<yn.cols()<<std::endl;


    Eigen::VectorXd d_x_y = w_pos * (x.transpose() - y).array().square().rowwise().sum() + w_nor * (xn.transpose() - yn).array().square().rowwise().sum();
    // std::cout<<d_x_y<<"d_x_y"<<std::endl;
    // exit(0);
    
    
    // std::cout<<"ytrewq4"<<std::endl;
    // sample new barycenters for this iteration
    Eigen::VectorXi i_F_s; Eigen::MatrixX3d b_s; Eigen::MatrixXd m_dof_per_s;
    generate_barycenters_on_mesh( i_F_s,  b_s, m_dof_per_s);
    // std::cout<<"ytrew5"<<std::endl;
    // evaluate new barycenters on new pose
    Eigen::MatrixX3d p_s;
    Eigen::MatrixX3d n_s;
    barycenters_to_mesh_positions(b_s, i_F_s, vnkap_->v_p, F, p_s);
    barycenters_to_mesh_normals(b_s, i_F_s, vnkap_->n_p, F, n_s);
    // std::cout<<"ytrewq6"<<std::endl;
    // for each x, compute closest points from the newly evaluated points
    Eigen::VectorXd i_p_s_per_x;
    Eigen::VectorXd d_x_p;
    compute_closest_point_on_mesh( p_s, n_s, i_p_s_per_x, d_x_p);
    // std::cout<<"ytrew7"<<std::endl;
    //i_p_s_per_x = np.array(i_p_s_per_x); d_x_p = np.array(d_x_p);

    // for each x, update the closest point if any of the new points are closer than previous points

    
    Eigen::ArrayXd m_p_over_y = (d_x_p.array() < d_x_y.array()).cast<double>();
    Eigen::ArrayXd one_minus_m_p_over_y = 1-m_p_over_y;
    ydata_new->y = p_s(i_p_s_per_x, Eigen::placeholders::all).array().colwise()*m_p_over_y+ydata_old->y.array().colwise()*one_minus_m_p_over_y;
    // //y_new = np.where(m_p_over_y[:, np.newaxis], p_s(i_p_s_per_x, Eigen::placeholders::all), y);
    ydata_new->yn = n_s(i_p_s_per_x, Eigen::placeholders::all).array().colwise()*m_p_over_y+ydata_old->yn.array().colwise()*one_minus_m_p_over_y;
    // //yn_new = np.where(m_p_over_y[:, np.newaxis], n_s(i_p_s_per_x, Eigen::placeholders::all), yn);
    //i_F_s(i_p_s_per_x).array()*m_p_over_y;
    ydata_new->i_F_y = i_F_s(i_p_s_per_x).array()*m_p_over_y.cast<int>()+(ydata_old->i_F_y).array()*one_minus_m_p_over_y.cast<int>();
    // //i_F_y_new = m_p_over_y.select(i_F_s(i_p_s_per_x), i_F_y);
    ydata_new->b_y = b_s(i_p_s_per_x, Eigen::placeholders::all).array().colwise()*m_p_over_y+ydata_old->b_y.array().colwise()*one_minus_m_p_over_y;
    // //b_y_new = np.where(m_p_over_y[:, np.newaxis], b_s(i_p_s_per_x, Eigen::placeholders::all), b_y)
    ydata_new->m_dof_per_y = m_dof_per_s(i_p_s_per_x, Eigen::placeholders::all).array().colwise()*m_p_over_y+ydata_old->m_dof_per_y.array().colwise()*one_minus_m_p_over_y;
    // // m_dof_per_y_new = np.where(m_p_over_y[:, np.newaxis], m_dof_per_s[i_p_s_per_x], m_dof_per_y)



    return; // y_new, yn_new, i_F_y_new, b_y_new, m_dof_per_y_new
}
