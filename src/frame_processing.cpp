#include<Reg.h>

void Reg::process_frame_(cv::Mat& color_raw, cv::Mat& depth_raw, cv::Mat& depth_proc2){
    cv::Mat mask_near_far = (depth_raw > d_near) & (depth_raw < d_far);

    
    cv::Mat depth_proc1, color_proc, color_proc1, color_proc2;  // Result output
    depth_proc1.setTo(0.0);              //is it required 
    
    depth_raw.copyTo(depth_proc1, mask_near_far);

    Eigen::MatrixXd depth_proc;
    cv2eigen(depth_proc1, depth_proc);


    color_proc1.setTo(0.0);                       //is it required 
    color_raw.copyTo(color_proc, mask_near_far);


    cvtColor(color_proc,color_proc2, cv::COLOR_RGB2HSV);
    medianBlur(color_proc2,color_proc1,5);

    inRange(color_proc1, cv::Scalar(50, 100, 0), cv::Scalar(120, 255, 255), color_proc2);

    cv::Mat labels, stats, centroids, label_ids_sorted_by_area;
    int n_labels = connectedComponentsWithStats(color_proc2, labels, stats, centroids);


    int h=depth_proc.rows(), w = depth_proc.cols();
    Eigen::MatrixXd V(h,w), U(h,w);
    auto temp1 = Eigen::RowVectorXd::LinSpaced(w, 0, w-1);
    auto temp2 = Eigen::VectorXd::LinSpaced(h, 0, h-1);
    
    U.rowwise() = temp1;
    V.colwise() = temp2;

    auto X = (U.array() - cx) * depth_proc.array() / fx;
    auto Y = (V.array() - cy) * depth_proc.array() / fy;
    auto Z = depth_proc.array();



    Eigen::MatrixX3d xyz_crop_center_new;
    double crop_radius = 200, wb_size = -10;


    cv::Mat mask_wb_dilated;
    Eigen::MatrixXd mask_wb, mask_wb_;

    if (n_labels < 2) {   // no component except background
        mask_wb_dilated = cv::Mat::zeros(color_proc2.rows, color_proc2.cols, CV_8U);
        cv2eigen(mask_wb_dilated,mask_wb);
        xyz_crop_center_new = xyz_crop_center;
    }
    else {
        cv::sortIdx(stats.col(4), label_ids_sorted_by_area, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

        auto label_wb_comp = label_ids_sorted_by_area.at<int>(1);

        cv::dilate(labels == label_wb_comp, mask_wb_dilated, cv::Mat::ones(5,5,CV_8U));


        cv::cv2eigen(mask_wb_dilated, mask_wb_);

        mask_wb  = mask_wb_.array()*(depth_proc.array()+0.0001)/255.0;
        Eigen::VectorXd I,J,V;
        igl::find(mask_wb,I,J,V);
        Eigen::MatrixX3d uvd_wb(I.size(),3);
        uvd_wb.col(0) = J;
        uvd_wb.col(1) = I;
        uvd_wb.col(2) = V;


        Eigen::MatrixX3d xyz_wb;
        uvd2xyz(uvd_wb, xyz_wb, fx, fy, cx, cy);
        Eigen::MatrixXd xyz_avg_wb = xyz_wb.colwise().mean();


        Eigen::MatrixXd dist_sq_from_wb = (X.array() - xyz_avg_wb(0,0)).square() + (Y.array() - xyz_avg_wb(0,1)).square() + (Z.array() - xyz_avg_wb(0,2)).square();
        double wrist_range_radius_sq = 40000.0;


        Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> mask_wrist_range = dist_sq_from_wb.array() < wrist_range_radius_sq;

        if ((mask_wrist_range).count() > 1000){    

            Eigen::VectorXd I,J,V;

            auto mask_wrist_range_ = mask_wrist_range.cast<double>().array()*(depth_proc.array()+0.0001);
            igl::find(mask_wrist_range_,I,J,V);
            Eigen::MatrixX3d uvd_wrist_range(I.rows(),3);
            uvd_wrist_range.col(0) = J;
            uvd_wrist_range.col(1) = I; 
            uvd_wrist_range.col(2) = V;

            Eigen::MatrixX3d xyz_wrist_range;
            uvd2xyz(uvd_wrist_range, xyz_wrist_range, fx, fy, cx, cy);

            Eigen::MatrixXd xyz_wrist_range_centered = xyz_wrist_range.rowwise() - xyz_wrist_range.colwise().mean();  //check
            Eigen::Matrix3d cov = (xyz_wrist_range_centered.adjoint() * xyz_wrist_range_centered) / double(xyz_wrist_range_centered.rows() - 1);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
            if (eigensolver.info() != Eigen::Success) {std::cout<<"eigen val fail"<<std::endl; int x; std::cin>>x; };
            Eigen::Vector3d evals = eigensolver.eigenvalues();
            Eigen::Matrix3d evecs = eigensolver.eigenvectors();


            // is there an efficient way of finding the largest eigenvalue only?


            // // sort in decreasing order of eigenvalue
            int max = 0;
            if (evals(1)>evals(max)){
                max = 1;
            }
            if (evals(2)>evals(max)){
                max = 2;
            }
            Eigen::RowVector3d dir_max_var = evecs.col(max).normalized().array();
            
      
            if (dir_max_var(1) > 0){
                dir_max_var = -dir_max_var;
            }


            xyz_crop_center = xyz_avg_wb.array()+ dir_max_var.array() * (crop_radius + wb_size);
        }

        Eigen::ArrayXXd dist_sq_from_crop_center = (X.array() - xyz_crop_center(0,0)).square() + (Y.array() - xyz_crop_center(0,1)).square() + (Z.array() - xyz_crop_center(0,2)).square();
    
        auto neg_mask_sphere = (crop_radius*crop_radius) > dist_sq_from_crop_center;
        
        Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> neg_mask_sil = neg_mask_sphere.cast<bool>().array() && !mask_wb_.cast<bool>().array();
        
        cv::Mat neg_mask_sil_;

        eigen2cv(neg_mask_sil, neg_mask_sil_);

        //not required
        //cv::Mat color_proc3;  
        //depth_proc2.setTo(0.0);             
        // color_proc3.setTo(0.0);
        
        depth_proc1.copyTo(depth_proc2, neg_mask_sil_);    //return depth_proc2

    }

    return;
}



void Reg::compute_sil_idx_at_each_pixel(cv::Mat& depth_proc){
    // a silhouette image is the binary image with 1 outside the hand region and 0 inside
    // at each pixel, find index to the closest pixel with value 1, using Distance Transform
    // the points are represented in image frame where origin is at top left; this is consistent with the perspective projection using intrinsic camera parameters
    //cv::Mat dst, labels;
    

    cv::Mat labels, channel0, channel1;
    cv::Mat dst;
    cv::Mat dd = depth_proc<=0;
    cv::distanceTransform(dd,dst,labels, cv::DIST_L2, cv::DIST_MASK_PRECISE,cv::DIST_LABEL_PIXEL); 


    Eigen::MatrixXi labels_;
    cv::cv2eigen(labels, labels_);

    label2x(0) = (-1);
    label2y(0) = (-1);
    int c = 1;


    for (int row = 0; row < dd.rows; ++row){
        for (int col = 0; col < dd.cols; ++col){
            if(dd.at<uchar>(row,col)==0){
                label2x(c) = col;
                label2y(c) = row;
                c+=1;
            }
        }
    }

    //std::cout<<labels_.rows()<<" "<<labels_.cols()<<std::endl;
    label1.resize( img_height, img_width);
    label0.resize( img_height, img_width);


    for (int i=0;i<img_height;i++){
        for (int j=0;j<img_width;j++){
            label1(i,j) = label2y(labels_(i,j));
            label0(i,j) = label2x(labels_(i,j));
        }
    }


    return;
}



void Reg::depth_to_point_cloud(cv::Mat& depth_proc){

    auto intrensic = open3d::camera::PinholeCameraIntrinsic(depth_proc.cols, depth_proc.rows, fx, fy, cx, cy);
    

    open3d::geometry::Image img;
    // Allocate data buffer  
    img.Prepare(depth_proc.cols, depth_proc.rows, 1, 2);
    cv::Mat depth_proc_;
    //depth_proc -=0.499999 ;
    depth_proc.convertTo(depth_proc_,CV_16U);
    // mat is your cv::Mat depth image
    memcpy(img.data_.data(), depth_proc_.data, img.data_.size());
    
    auto pcd = open3d::geometry::PointCloud::CreateFromDepthImage(img, intrensic);

    // std::uint16_t *p = reinterpret_cast<std::uint16_t*>(img.data_.data());
    // for (int i = 0; i < img.data_.size() / 2; p++, ++i) {
    //     std::cout << *p << " ";
    // }

    
    auto xdense_ = (*pcd).points_;

    x_dense.resize(3,xdense_.size());

 
    for (int i=0;i<xdense_.size();i++){
        x_dense.col(i) = xdense_[i];
    }

    if ((*pcd).points_.size()>n_x){
        pcd = std::get<0>((*pcd).RemoveRadiusOutliers(20,0.01));
    }

    if ((*pcd).points_.size() > n_x){
        std::vector<size_t> ids_to_choose;
        furthest_point_downsample_ids((*pcd).points_, n_x, ids_to_choose, x);
        pcd = (*pcd).SelectByIndex(ids_to_choose);
    }
    
    if ((*pcd).points_.size() > 0){
        (*pcd).EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(10));
        (*pcd).OrientNormalsTowardsCameraLocation();
    }
        
    auto x_ = (*pcd).points_;
    x.resize(3,x_.size());

 
    for (int i=0;i<x_.size();i++){
        x.col(i) = x_[i];
    }

    auto xn_ = (*pcd).normals_;

    xn.resize(3,x.cols());

    for (int i=0;i<x.cols();i++){
        xn.col(i) = xn_[i];
    }

    return;
}




void Reg::load_frame(int i){

    

    char name[] = "./../data/color_raw/00000.png";
    name[24] = i%10+48;
    name[23] = 48+(i%100)/10;
    name[22] = 48+(i/100)%10;
    cv::Mat color_raw_BGR= cv::imread(name), color_raw; 


    Eigen::MatrixXd depth;
    cvtColor(color_raw_BGR, color_raw, cv::COLOR_BGR2RGB);
    


    char name1[] = "./../data/depth_raw_data/d000.dat";
    name1[28] = 48+i%10;
    name1[27] = 48+(i%100)/10;
    name1[26] = 48+(i/100)%10;



    Eigen::read_binary(name1,depth);
 
    //std::cout<<depth.rows()<< " "<<depth.cols()<<" ";
    cv::Mat depth_raw;
    cv::eigen2cv(depth, depth_raw);

   

    cv::Mat depth_proc;


    process_frame_(color_raw, depth_raw, depth_proc);

    compute_sil_idx_at_each_pixel(depth_proc);


    depth_to_point_cloud(depth_proc);


}
