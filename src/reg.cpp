#include<Reg.h>



void Reg::solve_(Eigen::MatrixXd& JtJ, Eigen::VectorXd& Jte, double& w_damp, Eigen::VectorXd& dpose){
    //I_J = np.ix_(self.i_dof_update, self.i_dof_update)
    
    Eigen::VectorXd Jte_dof = Jte(i_dof_update);
    int n = i_dof_update.size();
    Eigen::MatrixXd JtJ_dof = JtJ(i_dof_update,i_dof_update) + w_damp * Eigen::MatrixXd::Identity(n,n);

    Eigen::VectorXd dpose_dof = JtJ_dof.colPivHouseholderQr().solve(Jte_dof);
    //dpose_dof = scipy.linalg.solve(JtJ_dof + w_damp_init * np.identity(n), Jte_dof, assume_a='pos')  // (|i_dof|,)
    dpose = Eigen::VectorXd::Zero(Jte.size());
    dpose(i_dof_update) = dpose_dof;

    return;

}


void Reg::generate_barycenters_on_mesh(Eigen::VectorXi& i_F_s,Eigen::MatrixX3d& b_s, Eigen::MatrixXd& m_dof_per_s){
    // these barycenters will be evaluated for different mesh pose and then projected onto the image
    // the projected points can then be used for computing the 2d data term


    sample_face_ids(i_f_per_part, n_f_per_part, F.rows(), n_s_approx, i_F_s);
    /*
    if (i_F_s.size() ==491){
    i_F_s<< 221,  237, 1487, 1506,   18 ,  78, 1491, 1506,   86,   89, 1502,  214,   87, 1491,
   86,  213,  357 , 360,   16,    2 ,1484 ,1476 ,1483,  360, 1482,   83,  327, 1481,
   23,   81,   23 ,1249, 1534,  236 ,1245 ,1246 ,1478, 1233, 1252, 1237,   17, 1335,
   92, 1238, 1252 ,1251,  235, 1534 ,1335 ,1532 , 233, 1353, 1295, 1321, 1374, 1293,
 1353, 1281, 1298 ,1379, 1309, 1319 ,1295 ,1276 ,1309, 1330, 1312, 1359, 1302, 1329,
 1358, 1374, 1309 ,1268, 1352, 1320 ,1369 ,1313 ,1303, 1270, 1312, 1339, 1358, 1367,
 1286,  263, 1451 , 107,  163, 1456 , 262 , 246 , 312,  246, 1486,  363,    8,   60,
  251, 1454,    9 , 313,  363, 1449 ,1459 ,  64 , 259, 1451,  190,   64,  243,   11,
  148,  523,  352 , 523,  522,  524 , 198 , 264 , 344,  156,  527,  100,  522,  346,
  155,  514,  250 , 522,  532,  184 , 264 , 204 , 385,  381,  456,  426,  467,  466,
  460,  157,   35 , 315,  289,  460 , 426 , 302 , 368,  369,  460,   29,  315,  384,
  305,  462,  507 , 494,  443,  500 , 400 , 476 , 421,  418,  478,  488,  501,  415,
  499,  495,  427 , 399,  489,  494 , 443 , 410 , 408,  433,  487,  427,  396,  480,
  486,  411,  453 , 435,  447,  392 , 473 , 420 , 333,  191,  765,   40,  181,  765,
  769,  181,   10 , 748,  575,  605 , 294 , 561 , 758,  757,  549,  601,  256,  753,
  546,  551,  764 , 578,  758,  539 , 750 ,1390 , 256,  542,  294,  571,  579,  538,
  690,  572,  692 , 620,  563,  615 , 621 , 557 , 557,  587,  537,  590,  594,  613,
  607,  592,  567 , 608,  592,  727 , 712 , 715 , 697,  666,  657,  676,  697,  666,
  640,  636,  629 , 646,  671,  657 , 644 , 660 , 714,  660,  736,  698,  639,  741,
  732,  720,  642 , 722,  712,  719 , 713 , 701 , 663,  736,  742,  711,  174,  189,
  280,  108,  109 , 188,  280,  144 , 151 , 119 ,1427,  189,  151,  377, 1526,  788,
 1388,  993,  995 , 832, 1391,  370 , 283 ,1398 , 814,  801,  990,  788,  984,  999,
  996,  295, 1398 , 991,  988,  779 ,1395 , 798 , 931,  807,  781,  892,  844,  824,
  936,  936,  773 , 847,  860,  852 , 845 , 836 , 836,  817,  796,  824,  852,  931,
  935,  803,  775 , 979,  946,  955 , 961 , 902 , 966,  943,  943,  904,  900,  904,
  950,  917,  865 , 863,  883,  916 , 938 , 977 , 957,  894,  918,  954,  968,  910,
  950,  885,  912 , 970,  904,  863 , 958 , 896 , 910, 1525,  241,  167,  161, 1495,
  140, 1410,  161 ,1446, 1432, 1413 , 291 , 228 ,1518, 1440,  241,   31,   74,  319,
  322,  239, 1522 , 325, 1445,   74 , 239 , 141 ,  76, 1404,   76,  241, 1423, 1444,
 1403, 1410,  171 ,1496,  115, 1437 ,  51 ,1410 , 165, 1025, 1033, 1225, 1225, 1023,
 1063, 1008, 1393 ,1433, 1034, 1063 ,1063 ,1224 ,1008, 1054, 1228, 1214, 1070,  371,
 1429, 1039, 1059 ,1090, 1071, 1082 ,1083 ,1032 ,1002, 1173, 1159, 1032, 1089, 1077,
 1116, 1031, 1092 ,1093, 1182, 1050 ,1067 ,1028 ,1002, 1004, 1068, 1035, 1188, 1113,
 1171, 1150, 1129 ,1141, 1212, 1208 ,1122 ,1208 ,1117, 1193, 1130, 1179, 1177, 1197,
 1133, 1207, 1201 ,1097, 1104, 1106 ,1176 ,1131 ,1109, 1104, 1204, 1156, 1098, 1121,
 1177;

    }
    if (i_F_s.size() ==200){
        i_F_s<< 87, 1505, 1505, 1499, 1503, 1505,  222,  327, 1468,  334,  360,   16,    0, 1245,
 1530, 1241,  331, 1334, 1255, 1258, 1331, 1346, 1360, 1324, 1322, 1377, 1339, 1292,
 1364, 1357, 1339, 1300, 1345, 1363, 1367,  251,   11,  246,  248,  316,    6, 1486,
   60, 1449,    6, 1486,  524,  308,  200,  265,  287,  134,  519,  344,  385,  157,
  462,  350,  315,   35,  386,  456,  458,  196,  404,  416,  444,  401,  402,  436,
  453,  395,  425,  450,  496,  468,  444,  406,   66,  270,  772,  751,  561,  583,
  142,  561,  546,  752,  555,  546,  534,  591,  557,  590,  574,  587,  581,  543,
  591,  700,  659,  637,  661,  630,  677,  722,  665,  708,  717,  736,  721,  742,
  626,  109,  127,  118,  109,  118,  174,  841,  795,  232,  992,  837,  990, 1000,
 1387,  812,  851,  808,  845,  855,  937,  820,  848,  804,  824,  929,  896,  910,
  866,  865,  975,  885,  954,  879,  903,  873,  967,  960,  955,  972,  323,   32,
 1519,  165, 1511, 1442,  318,   71,  291, 1447,   77, 1406,   69, 1516,  227, 1521,
  140,  126, 1008, 1025, 1222, 1051, 1433, 1216, 1218, 1088, 1168, 1062, 1164, 1002,
 1088, 1071, 1078, 1162, 1060, 1134, 1146, 1144, 1128, 1194, 1111, 1202, 1146, 1175,
 1178, 1145, 1154, 1130;
    }*/
    generate_random_barycentric_coordinates(i_F_s.size(), b_s);
    //std::cout<<b_s<<std::endl;
    m_dof_per_s = m_dof_per_face(i_F_s, Eigen::placeholders::all).cast<double>(); // (|b|, 26)

    return;
}


Reg::Reg(){
    vnkap_orig = new vnkap;
    vnkap_new = new vnkap;
    theta_orig = new theta;
    theta_new = new theta;
    y_data_orig = new y_data;
    y_data_new = new y_data;

    xpq_orig = new xpq;
    xpq_new = new xpq;

 
    set_params();
   
    set_Amano_vars();

    set_RegistrationBase_var();

    set_Registration_vars();


}

Reg::~Reg(){
    delete vnkap_orig;
    delete vnkap_new;
    delete theta_orig;
    delete theta_new;
    delete y_data_orig;
    delete y_data_new;
    delete xpq_new;
    delete xpq_orig;
}


Eigen::Matrix3d rotvec2mat(const Eigen::Vector3d& rvec) {
    double angle = rvec.norm(); // angle of rotation
    if (angle < 1e-8) {
        return Eigen::Matrix3d::Identity(); // no rotation, return identity matrix
    }
    Eigen::Vector3d axis = rvec / angle; // axis of rotation
    Eigen::AngleAxisd aa(angle, axis); // angle-axis representation
    return aa.toRotationMatrix(); // convert angle-axis to rotation matrix
}


// v_art slightly off due to manoP->v_s->rp->tp->v_art
void Reg::deform(theta* theta_,Eigen::MatrixX3d& v_art, Eigen::MatrixXd& Rs_rel_art, Eigen::MatrixXd& Rs,  Eigen::MatrixX3d& a_prime){
        

    Rs_rel_art.resize(theta_->theta.rows(),9);

    Eigen::MatrixX3d tttemp = axes.array().colwise()*(theta_->theta).array();

    // #pragma omp parallel for
    for(int iii=0;iii<Rs_rel_art.rows();iii++){
        Eigen::Vector3d rvec = tttemp.row(iii);
        Rs_rel_art.row(iii) = rotvec2mat(rvec).reshaped<Eigen::RowMajor>();
    }




    // Eigen::ArrayXd sin_theta = (theta_->theta).array().sin();
    // Eigen::ArrayXd cos_theta = (theta_->theta).array().cos();
    // Eigen::ArrayXd one_minus_cos_theta = 1 - cos_theta;

    // Eigen::ArrayXd x = axes.col(0).array(), y = axes.col(1).array(), z = axes.col(1).array();

    // Eigen::ArrayXd zs = z*sin_theta, xs = x*sin_theta, ys = y*sin_theta;

    // Eigen::ArrayXd tx = one_minus_cos_theta*x, ty = one_minus_cos_theta*y;
    // Eigen::ArrayXd txx = tx*x, txy = tx*y, txz = tx*z, tyz = ty*z, tyy = ty*y, tzz = one_minus_cos_theta*z*z;
    // Rs_rel_art.col(0) = txx + cos_theta;
    // Rs_rel_art.col(1) = txy - zs;
    // Rs_rel_art.col(2) = txz + ys;
    // Rs_rel_art.col(3) = txy + zs;
    // Rs_rel_art.col(4) = tyy + cos_theta;
    // Rs_rel_art.col(5) = tyz - xs;
    // Rs_rel_art.col(6) = txz - ys;
    // Rs_rel_art.col(7) = tyz + xs;
    // Rs_rel_art.col(8) = tzz + cos_theta;

    Eigen::MatrixX3d v_s = v + (mano_S * beta).reshaped(3,v.rows()).transpose();    //isnt it constant> 
    // std::cout<<v_s<<std::endl<<std::endl;
    // std::cout<<v_s<<std::endl;
    // exit(0);


    Eigen::MatrixX3d k_s = K * v_s;

    Eigen::MatrixXd Rs_rel_art_wrist = Rs_rel_art(Eigen::seq(0,19,4),Eigen::placeholders::all);
    Eigen::MatrixXd Rs_rel_art_mcp = Rs_rel_art(Eigen::seq(1,19,4),Eigen::placeholders::all);
    Eigen::MatrixXd Rs_rel_art_pip = Rs_rel_art(Eigen::seq(2,19,4),Eigen::placeholders::all);
    Eigen::MatrixXd Rs_rel_art_dip = Rs_rel_art(Eigen::seq(3,19,4),Eigen::placeholders::all);


    Eigen::MatrixXd Rs_rel_per_joint(15,9);
    Rs_rel_per_joint(Eigen::seq(1,14,3),Eigen::placeholders::all) = Rs_rel_art_pip;
    Rs_rel_per_joint(Eigen::seq(2,14,3),Eigen::placeholders::all) = Rs_rel_art_dip;
    
    Rs_rel_per_joint(Eigen::seq(0,14,3),Eigen::placeholders::all) = compose_mat(Rs_rel_art_wrist, Rs_rel_art_mcp);

    Eigen::RowVectorXd I = Eigen::RowVectorXd::Zero(9);
    I(0) = 1.0;
    I(4) = 1.0; 
    I(8) = 1.0;

    //v_s is slightly off.

    // std::cout<<v_s<<std::endl<<std::endl;
    v_s += (mano_P * ((Rs_rel_per_joint(i_Rs_rel_art_mano, Eigen::placeholders::all)).rowwise()-I).reshaped<Eigen::RowMajor>()).reshaped<Eigen::RowMajor>(v.rows(),3);

    


    Rs.resize(20,9);//np.empty((20,3,3))

    Eigen::MatrixXd Rs_wrist = Eigen::MatrixXd::Zero(5, 9);        
    Rs_wrist(Eigen::placeholders::all,{0,4,8}) = Eigen::MatrixXd::Ones(5, 3);                                                                
    Eigen::MatrixXd Rs_mcp = compose_mat(Rs_rel_art_wrist, Rs_rel_art_mcp);
    Eigen::MatrixXd Rs_pip = compose_mat(Rs_mcp, Rs_rel_art_pip);        
    Eigen::MatrixXd Rs_dip = compose_mat(Rs_pip, Rs_rel_art_dip);        

    Rs(Eigen::seq(0,19,4),Eigen::placeholders::all) = Rs_wrist; 
    Rs(Eigen::seq(1,19,4),Eigen::placeholders::all) = Rs_mcp  ;
    Rs(Eigen::seq(2,19,4),Eigen::placeholders::all) = Rs_pip  ;
    Rs(Eigen::seq(3,19,4),Eigen::placeholders::all) = Rs_dip  ;
    


    Eigen::MatrixX3d b = k_s(Eigen::seq(1,20), Eigen::placeholders::all);
    Eigen::MatrixX3d a = k_s(Eigen::seq(0,19), Eigen::placeholders::all);
    a(Eigen::seq(4,19,4), Eigen::placeholders::all).rowwise() = k_s.row(0);



    Eigen::VectorXd phi_i_wrist = phi(Eigen::seq(0,19,4));
    Eigen::VectorXd phi_i_mcp = phi(Eigen::seq(1,19,4));
    Eigen::VectorXd phi_i_pip = phi(Eigen::seq(2,19,4));
    Eigen::VectorXd phi_i_dip = phi(Eigen::seq(3,19,4));


    Eigen::MatrixX3d a_i_wrist = a(Eigen::seq(0,19,4),Eigen::placeholders::all);
    Eigen::MatrixX3d a_i_mcp = a(Eigen::seq(1,19,4),Eigen::placeholders::all);
    Eigen::MatrixX3d a_i_pip = a(Eigen::seq(2,19,4),Eigen::placeholders::all);
    Eigen::MatrixX3d a_i_dip = a(Eigen::seq(3,19,4),Eigen::placeholders::all);

    a_prime.resize(20,3);
    Eigen::MatrixX3d a_prime_wrist(5,3);
    a_prime_wrist.rowwise() = a.row(0);
    // // a_prime_mcp = np.einsum("i,ijk,ik->ij", phi[0:20:4], Rs[0:20:4], a[1:20:4] - a[0:20:4]) + a_prime[0:20:4]
    // // a_prime_pip = np.einsum("i,ijk,ik->ij", phi[1:20:4], Rs[1:20:4], a[2:20:4] - a[1:20:4]) + a_prime[1:20:4]
    // // a_prime_dip = np.einsum("i,ijk,ik->ij", phi[2:20:4], Rs[2:20:4], a[3:20:4] - a[2:20:4]) + a_prime[2:20:4]
    
    Eigen::MatrixX3d temp = ((a_i_mcp - a_i_wrist).array().colwise()*phi_i_wrist.array()).matrix();
    Eigen::MatrixX3d a_prime_mcp = mat_rotate(Rs_wrist, temp) + a_prime_wrist;
    temp = ((a_i_pip - a_i_mcp).array().colwise()*phi_i_mcp.array()).matrix();
    Eigen::MatrixX3d a_prime_pip = mat_rotate(Rs_mcp, temp) + a_prime_mcp;
    temp = ((a_i_dip - a_i_pip).array().colwise()*phi_i_pip.array()).matrix();
    Eigen::MatrixX3d a_prime_dip = mat_rotate(Rs_pip, temp) + a_prime_pip;
    
    a_prime(Eigen::seq(0,19,4),Eigen::placeholders::all) = a_prime_wrist;
    a_prime(Eigen::seq(1,19,4),Eigen::placeholders::all) = a_prime_mcp;
    a_prime(Eigen::seq(2,19,4),Eigen::placeholders::all) = a_prime_pip;
    a_prime(Eigen::seq(3,19,4),Eigen::placeholders::all) = a_prime_dip;


    Eigen::MatrixX3d R1 = Rs(Eigen::placeholders::all,Eigen::seq(0,2));
    Eigen::MatrixX3d R2 = Rs(Eigen::placeholders::all,Eigen::seq(3,5));
    Eigen::MatrixX3d R3 = Rs(Eigen::placeholders::all,Eigen::seq(6,8));

    Eigen::Matrix3Xd v_s_t = v_s.transpose();
    Eigen::MatrixXd Rp1 = R1 * v_s_t;
    Eigen::MatrixXd Rp2 = R2 * v_s_t;
    Eigen::MatrixXd Rp3 = R3 * v_s_t;

    Eigen::VectorXd Ra1 = R1.cwiseProduct(a).rowwise().sum();
    Eigen::VectorXd Ra2 = R2.cwiseProduct(a).rowwise().sum();
    Eigen::VectorXd Ra3 = R3.cwiseProduct(a).rowwise().sum();

    Eigen::MatrixX3d s = (b-a).array().colwise()*(phi.array() - 1.0);

    Eigen::MatrixX3d Rs_ = mat_rotate(Rs,s);

    //std::cout<< W_endpoint.rows()<<" "<< W_endpoint.cols()<<" "<< Rs_.rows()<<" "<<Rs_.cols()<<std::endl;
    // W_endpoint.row(0).array()*Rs_.col(0).array().transpose();
    Eigen::MatrixXd eRs1 = W_endpoint.array().rowwise()*Rs_.col(0).array().transpose();
    Eigen::MatrixXd eRs2 = W_endpoint.array().rowwise()*Rs_.col(1).array().transpose();
    Eigen::MatrixXd eRs3 = W_endpoint.array().rowwise()*Rs_.col(2).array().transpose();

    //(Ra1-a_prime.col(0));
    //std::cout<<eRs1.rows()<<" "<<eRs1.cols()<<" "<<Rp1.rows()<<" "<<Rp1.cols()<<" "<<Ra1.size()<<std::endl;
    //(eRs1+Rp1).array().row(0);
    Eigen::MatrixXd Tp1 = (eRs1+Rp1.transpose()).array().rowwise()+(a_prime.col(0)-Ra1).array().transpose();
    Eigen::MatrixXd Tp2 = (eRs2+Rp2.transpose()).array().rowwise()+(a_prime.col(1)-Ra2).array().transpose();
    Eigen::MatrixXd Tp3 = (eRs3+Rp3.transpose()).array().rowwise()+(a_prime.col(2)-Ra3).array().transpose();
    
    v_art.resize(v.rows(),3);
    v_art.col(0) = Tp1.cwiseProduct(W_bone).rowwise().sum().transpose();
    v_art.col(1) = Tp2.cwiseProduct(W_bone).rowwise().sum().transpose();
    v_art.col(2) = Tp3.cwiseProduct(W_bone).rowwise().sum().transpose();



    return;
}




//correct
void Reg::compute_axis_per_dof_in_world_space( Eigen::Matrix3d& R_glob_ref_x, Eigen::Matrix3d& R_glob_ref_y, Eigen::Matrix3d& R_glob_ref_z, Eigen::MatrixXd& Rs_rel, vnkap* vnkap_){
    
    Eigen::Matrix3d init_ref_x = R_glob_init * R_glob_ref_x;
    Eigen::Matrix3d init_ref_x_ref_y = R_glob_init * R_glob_ref_x * R_glob_ref_y;
    Eigen::Matrix3d R_glob = (init_ref_x_ref_y * R_glob_ref_z);
    
    Eigen::MatrixXd Rs_rel1 = Rs_rel(Eigen::seq(0,19,4), Eigen::placeholders::all);
    Eigen::MatrixXd Rs_rel2 = Rs_rel(Eigen::seq(1,19,4), Eigen::placeholders::all);
    Eigen::MatrixXd Rs_rel3 = Rs_rel(Eigen::seq(2,19,4), Eigen::placeholders::all);
    Eigen::MatrixXd temp1; 
    Eigen::MatrixXd temp2; 
    Eigen::MatrixXd temp3; 

    compose_mat(R_glob, Rs_rel1, temp1);
    compose_mat(temp1, Rs_rel2, temp2);
    compose_mat(temp2, Rs_rel3, temp3);


    vnkap_->axis_per_dof.resize(23,3);

    vnkap_->axis_per_dof.row(0) = R_glob_init.col(0);
    vnkap_->axis_per_dof.row(1) = init_ref_x.col(1);
    vnkap_->axis_per_dof.row(2) = init_ref_x_ref_y.col(2);

    Eigen::MatrixX3d v1=axes({1,5,9,13,17},Eigen::placeholders::all), v2=axes({2,6,10,14,18},Eigen::placeholders::all), v3=axes({3,7,11,15,19},Eigen::placeholders::all);
    vnkap_->axis_per_dof({3,7,11,15,19},Eigen::placeholders::all) = axes({0,4,8,12,16},Eigen::placeholders::all)*R_glob.transpose();
    vnkap_->axis_per_dof({4,8,12,16,20},Eigen::placeholders::all) = mat_rotate(temp1,v1);
    vnkap_->axis_per_dof({5,9,13,17,21},Eigen::placeholders::all) = mat_rotate(temp2, v2);
    vnkap_->axis_per_dof({6,10,14,18,22},Eigen::placeholders::all) = mat_rotate(temp3, v3);


    return;
}


//correct
void Reg::compute_pivot_per_dof_in_world_space(Eigen::Matrix3d& R_glob, Eigen::Vector3d& t_glob, Eigen::MatrixX3d& a_prime, vnkap* vnkap_){
    
    //std::cout<<a_prime.rows()<<" "<<a_prime.cols()<<" "<<R_glob.rows()<<" "<<R_glob.cols()<<" "<<t_glob.size()<<std::endl;
    Eigen::MatrixX3d a_prime_after_glob = (a_prime * R_glob.transpose()).rowwise() + t_glob.transpose();    // (20, 3)
    vnkap_->pivot_per_dof.resize(23,3);
    vnkap_->pivot_per_dof.row(0) = t_glob;
    vnkap_->pivot_per_dof.row(1) = t_glob;
    vnkap_->pivot_per_dof.row(2) = t_glob;
    vnkap_->pivot_per_dof({3,7,11,15,19},Eigen::placeholders::all) = a_prime_after_glob({1,5,9,13,17}, Eigen::placeholders::all);
    vnkap_->pivot_per_dof({4,8,12,16,20}, Eigen::placeholders::all) = a_prime_after_glob({1,5,9,13,17}, Eigen::placeholders::all);
    vnkap_->pivot_per_dof({5,9,13,17,21}, Eigen::placeholders::all) = a_prime_after_glob({2,6,10,14,18}, Eigen::placeholders::all);
    vnkap_->pivot_per_dof({6,10,14,18,22}, Eigen::placeholders::all) = a_prime_after_glob({3,7,11,15,19}, Eigen::placeholders::all);

    return;
}



//check n_F, otherwise correct
void Reg::deform_and_compute_linearized_info(vnkap * vnkap_, theta * theta_){
    // shape and articualate
    Eigen::MatrixX3d v_art;
    Eigen::MatrixXd Rs_rel_art;
    Eigen::MatrixXd Rs;
    Eigen::MatrixX3d a_prime;

    deform(theta_,v_art, Rs_rel_art, Rs, a_prime);
    //std::cout<<Rs_rel_art<<std::endl<<std::endl;

    // global transform
    Eigen::Vector3d cos_theta =  theta_->theta_glob.array().cos();
    Eigen::Vector3d sin_theta =  theta_->theta_glob.array().sin();
    Eigen::Matrix3d R_glob_ref_x = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d R_glob_ref_y = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d R_glob_ref_z = Eigen::Matrix3d::Zero();

    R_glob_ref_x(0,0) = 1.0;
    R_glob_ref_x(1,1) = cos_theta(0);
    R_glob_ref_x(2,2) = cos_theta(0);
    R_glob_ref_x(1,2) = -sin_theta(0);
    R_glob_ref_x(2,1) = sin_theta(0);

    R_glob_ref_z(2,2) = 1.0;
    R_glob_ref_z(0,0) = cos_theta(2);
    R_glob_ref_z(1,1) = cos_theta(2);
    R_glob_ref_z(0,1) = -sin_theta(2);
    R_glob_ref_z(1,0) = sin_theta(2);

    R_glob_ref_y(1,1) = 1.0;
    R_glob_ref_y(0,0) = cos_theta(1);
    R_glob_ref_y(2,2) = cos_theta(1);
    R_glob_ref_y(0,2) = sin_theta(1);
    R_glob_ref_y(2,0) = -sin_theta(1);

    Eigen::Matrix3d R_glob = R_glob_init * R_glob_ref_x * R_glob_ref_y * R_glob_ref_z;
    vnkap_->v_p = (v_art * R_glob.transpose()).rowwise() + theta_->t_glob.transpose();



    compute_vertex_normals(vnkap_->v_p, F, vnkap_->n_p);



    vnkap_->k_p = K * vnkap_->v_p;

    // linearized info required for computing derivatives of energy

    compute_axis_per_dof_in_world_space(R_glob_ref_x, R_glob_ref_y, R_glob_ref_z, Rs_rel_art, vnkap_);
    compute_pivot_per_dof_in_world_space(R_glob,theta_->t_glob,a_prime, vnkap_);   // (26, 3)

    //std::cout<<(vnkap_->v_p)<<std::endl<<std::endl;
    // std::cout<<(vnkap_->n_p)<<std::endl;
    // std::cout<<"Qwer"<<std::endl;
    // exit(0);

}



void Reg::register_to_keypoints(){
    Eigen::MatrixXd JtJ; Eigen::VectorXd Jte; double E;
    
    compute_energy_for_keypoints(vnkap_orig, theta_orig, JtJ, Jte, E);

    double w_damp = w_damp_init;
    for (int iter=0; iter<n_iter; iter++){
        Eigen::VectorXd dpose;

        solve_(JtJ, Jte, w_damp, dpose);



        Eigen::Vector3d dt_glob = dpose(Eigen::seq(0,2));              theta_new->t_glob = theta_orig->t_glob + dt_glob;
        Eigen::Vector3d dtheta_glob = dpose(Eigen::seq(3,5));          theta_new->theta_glob = theta_orig->theta_glob + dtheta_glob;
        Eigen::VectorXd dtheta = dpose(Eigen::seq(6,dpose.size()-1));  theta_new->theta = theta_orig->theta + dtheta; 
        
        theta_new->theta = theta_new->theta.cwiseMax(theta_min).cwiseMin(theta_max);



        Eigen::MatrixXd JtJ_new; Eigen::VectorXd Jte_new; double E_new;
        deform_and_compute_linearized_info(vnkap_new, theta_new);

        compute_energy_for_keypoints( vnkap_new, theta_new, JtJ_new, Jte_new, E_new);

        if (E_new < E){   // iteration successful   
            w_damp *= 0.8;
            JtJ = JtJ_new;        Jte = Jte_new;                   E = E_new;
            delete theta_orig;                      delete vnkap_orig;     
            theta_orig = theta_new;                 vnkap_orig = vnkap_new; 
            theta_new = new theta;                  vnkap_new = new vnkap;
        }
        else{   // iteration unsuccessful
            w_damp *= 10;
        }
    }

}



void Reg::register_to_pointcloud(){
 
    
    compute_3d_correspondences(vnkap_orig, y_data_orig);


    compute_2d_correspondences(vnkap_orig, xpq_orig);

    

    Eigen::MatrixXd JtJ; Eigen::VectorXd Jte; double E;


    compute_energy_for_pointcloud( vnkap_orig, theta_orig, JtJ, Jte, E);


    double w_damp = w_damp_init;
    for (int iter=0; iter<n_iter; iter++){   
        Eigen::VectorXd dpose;

        solve_(JtJ, Jte, w_damp, dpose);

        Eigen::Vector3d dt_glob = dpose(Eigen::seq(0,2));              theta_new->t_glob = theta_orig->t_glob + dt_glob;
        Eigen::Vector3d dtheta_glob = dpose(Eigen::seq(3,5));          theta_new->theta_glob = theta_orig->theta_glob + dtheta_glob;
        Eigen::VectorXd dtheta = dpose(Eigen::seq(6,dpose.size()-1));  theta_new->theta = theta_orig->theta + dtheta; 

        theta_new->theta = theta_new->theta.cwiseMax(theta_min).cwiseMin(theta_max);


        Eigen::MatrixXd JtJ_new; Eigen::VectorXd Jte_new; double E_new;

        
        deform_and_compute_linearized_info(vnkap_new, theta_new);

       
        update_3d_correspondences(vnkap_new, y_data_orig, y_data_new);


        compute_2d_correspondences(vnkap_new, xpq_new);

        


        compute_energy_for_pointcloud( vnkap_new, theta_new, JtJ_new, Jte_new, E_new);


        if (E_new < E){   // iteration successful    
            w_damp *= 0.8;
            JtJ = JtJ_new;        Jte = Jte_new;                   E = E_new;
            delete theta_orig;                      delete vnkap_orig;     
            theta_orig = theta_new;                 vnkap_orig = vnkap_new; 
            theta_new = new theta;                  vnkap_new = new vnkap;

            delete y_data_orig;                      delete xpq_orig;     
            y_data_orig = y_data_new;                xpq_orig = xpq_new; 
            y_data_new = new y_data;                  xpq_new = new xpq;
        }
        else{   // iteration unsuccessful
            w_damp *= 10;
        }
    }

}





void Reg::start_engine(){
 


    char* name = new char[10];
    name[0] = 'v';
    name[4] = '.';
    name[5] = 't';
    name[6] = 'x';
    name[7] = 't';
    name[8] = 0;

    load_frame(0);


    register_to_keypoints();
    Eigen::Vector3d cos_theta =  theta_orig->theta_glob.array().cos();
    Eigen::Vector3d sin_theta =  theta_orig->theta_glob.array().sin();
    Eigen::Matrix3d R_glob_ref_x = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d R_glob_ref_y = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d R_glob_ref_z = Eigen::Matrix3d::Zero();

    R_glob_ref_x(0,0) = 1.0;
    R_glob_ref_x(1,1) = cos_theta(0);
    R_glob_ref_x(2,2) = cos_theta(0);
    R_glob_ref_x(1,2) = -sin_theta(0);
    R_glob_ref_x(2,1) = sin_theta(0);

    R_glob_ref_z(2,2) = 1.0;
    R_glob_ref_z(0,0) = cos_theta(2);
    R_glob_ref_z(1,1) = cos_theta(2);
    R_glob_ref_z(0,1) = -sin_theta(2);
    R_glob_ref_z(1,0) = sin_theta(2);

    R_glob_ref_y(1,1) = 1.0;
    R_glob_ref_y(0,0) = cos_theta(1);
    R_glob_ref_y(2,2) = cos_theta(1);
    R_glob_ref_y(0,2) = sin_theta(1);
    R_glob_ref_y(2,0) = -sin_theta(1);

    Eigen::Matrix3d R_glob = R_glob_init * R_glob_ref_x * R_glob_ref_y * R_glob_ref_z;
    R_glob_init = R_glob;

    theta_orig->theta_glob = Eigen::Vector3d::Zero();

    k_p_prev = NULL;

    deform_and_compute_linearized_info(vnkap_orig, theta_orig);


    register_to_pointcloud();


    delete __k_marked;
    __k_marked = NULL;





    // #pragma omp parallel for
    for (int i=1;i<100;i++){
        std::cout<<"frame "<<i<<std::endl;
        load_frame(i);
        
        name[1] = 48+i/100;
        name[2] = 48+(i%100)/10;
        name[3] = 48+i%10;
        std::ofstream MyFile(name);

        Eigen::Vector3d cos_theta =  theta_orig->theta_glob.array().cos();
        Eigen::Vector3d sin_theta =  theta_orig->theta_glob.array().sin();
        Eigen::Matrix3d R_glob_ref_x = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d R_glob_ref_y = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d R_glob_ref_z = Eigen::Matrix3d::Zero();

        R_glob_ref_x(0,0) = 1.0;
        R_glob_ref_x(1,1) = cos_theta(0);
        R_glob_ref_x(2,2) = cos_theta(0);
        R_glob_ref_x(1,2) = -sin_theta(0);
        R_glob_ref_x(2,1) = sin_theta(0);

        R_glob_ref_z(2,2) = 1.0;
        R_glob_ref_z(0,0) = cos_theta(2);
        R_glob_ref_z(1,1) = cos_theta(2);
        R_glob_ref_z(0,1) = -sin_theta(2);
        R_glob_ref_z(1,0) = sin_theta(2);

        R_glob_ref_y(1,1) = 1.0;
        R_glob_ref_y(0,0) = cos_theta(1);
        R_glob_ref_y(2,2) = cos_theta(1);
        R_glob_ref_y(0,2) = sin_theta(1);
        R_glob_ref_y(2,0) = -sin_theta(1);

        R_glob = R_glob_init * R_glob_ref_x * R_glob_ref_y * R_glob_ref_z;
        
        R_glob_init = R_glob; // use this as initial global transformation for next frame

        // /// since theta_glob captures offset from initial global transform, init to zero
        theta_orig->theta_glob = Eigen::Vector3d::Zero();
        // // t_glob, theta, beta are initialized using previous frame's estimates

        delete k_p_prev;

        k_p_prev = new Eigen::MatrixX3d;
        *k_p_prev = vnkap_orig->k_p;
        // // there are no keypoints for other frames
        
     
        register_to_pointcloud();

        
        MyFile<<vnkap_orig->v_p.reshaped<Eigen::RowMajor>()<<std::endl;
        MyFile.close();

    }

}