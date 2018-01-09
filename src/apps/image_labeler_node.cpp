#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <Eigen/Core>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <lucrezio_logical_camera/LogicalImage.h>
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace message_filters;
using namespace lucrezio_logical_camera;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


class ImageLabeler{
public:
  ImageLabeler(std::string robotname_ = ""):
    _robotname(robotname_),
    _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
    _depth_cloud_sub(_nh,"/camera/depth/points",1),
    _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
    _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_cloud_sub,_rgb_image_sub),
    _it(_nh){

    _got_info = false;
    _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
					    1000,
					    &ImageLabeler::cameraInfoCallback,
					    this);

    _synchronizer.registerCallback(boost::bind(&ImageLabeler::filterCallback, this, _1, _2, _3));

    _seq = 1;

    _label_image_pub = _it.advertise("/camera/rgb/label_image", 1);
    
    ROS_INFO("Starting training set generator node!");
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg){
    sensor_msgs::CameraInfo camerainfo;
    camerainfo.K = camera_info_msg->K;

    ROS_INFO("Got camera info!");
    _K(0,0) = camerainfo.K.c_array()[0];
    _K(0,1) = camerainfo.K.c_array()[1];
    _K(0,2) = camerainfo.K.c_array()[2];
    _K(1,0) = camerainfo.K.c_array()[3];
    _K(1,1) = camerainfo.K.c_array()[4];
    _K(1,2) = camerainfo.K.c_array()[5];
    _K(2,0) = camerainfo.K.c_array()[6];
    _K(2,1) = camerainfo.K.c_array()[7];
    _K(2,2) = camerainfo.K.c_array()[8];

    cerr << _K << endl;

    _got_info = true;
    _camera_info_sub.shutdown();
  }

  void filterCallback(const LogicalImage::ConstPtr& logical_image_msg,
		      const PointCloud::ConstPtr& scene_cloud_msg,
		      const sensor_msgs::Image::ConstPtr& rgb_image_msg){
    if(_got_info){

      cv_bridge::CvImageConstPtr rgb_cv_ptr;
      try{
	rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
      } catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }

      cv::Mat image = rgb_cv_ptr->image.clone();
      
      tf::StampedTransform depth_camera_pose;
      try {
	_listener.waitForTransform("map",
				   "camera_depth_optical_frame",
				   ros::Time(0),
				   ros::Duration(3));
	_listener.lookupTransform("map",
				  "camera_depth_optical_frame",
				  ros::Time(0),
				  depth_camera_pose);
      }
      catch(tf::TransformException ex) {
	ROS_ERROR("%s", ex.what());
      }

      Eigen::Isometry3f depth_camera_transform = tfTransform2eigen(depth_camera_pose);
      PointCloud::Ptr map_cloud (new PointCloud ());
      pcl::transformPointCloud (*scene_cloud_msg, *map_cloud, depth_camera_transform);
      map_cloud->header.frame_id = "/map";
      map_cloud->width  = scene_cloud_msg->width;
      map_cloud->height = scene_cloud_msg->height;
      map_cloud->is_dense = false;
        
      PointCloud::Ptr objects_cloud_msg (new PointCloud ());

      PointCloud::Ptr boxes_cloud_msg (new PointCloud ());
       
      tf::StampedTransform logical_camera_pose;
      tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

      for(int i=0; i < logical_image_msg->models.size(); i++){

	Eigen::Vector3f box_min (logical_image_msg->models.at(i).min.x,
				 logical_image_msg->models.at(i).min.y,
				 logical_image_msg->models.at(i).min.z);

	Eigen::Vector3f box_max (logical_image_msg->models.at(i).max.x,
				 logical_image_msg->models.at(i).max.y,
				 logical_image_msg->models.at(i).max.z);

	float x_range = box_max.x()-box_min.x();
	float y_range = box_max.y()-box_min.y();
	float z_range = box_max.z()-box_min.z();

	PointCloud::Ptr model_cloud (new PointCloud ());
	for(int k=0; k <= 1; k++)
	  for(int j=0; j <= 1; j++)
	    for(int i=0; i <= 1; i++){
	      model_cloud->points.push_back (pcl::PointXYZ(box_min.x() + i*x_range,
							   box_min.y() + j*y_range,
							   box_min.z() + k*z_range));
	    }

	PointCloud::Ptr transformed_model_cloud (new PointCloud ());
	tf::Transform model_pose;
	tf::poseMsgToTF(logical_image_msg->models.at(i).pose,model_pose);
	Eigen::Isometry3f model_transform = tfTransform2eigen(logical_camera_pose)*tfTransform2eigen(model_pose);
	pcl::transformPointCloud (*model_cloud, *transformed_model_cloud, model_transform);

	*boxes_cloud_msg += *transformed_model_cloud;
	    
	pcl::PointXYZ min_pt,max_pt;
	pcl::getMinMax3D(*transformed_model_cloud,min_pt,max_pt);
	    
	PointCloud::Ptr cloud_filtered_x (new PointCloud ());
	PointCloud::Ptr cloud_filtered_xy (new PointCloud ());
	PointCloud::Ptr cloud_filtered_xyz (new PointCloud ());

	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (map_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (min_pt.x,max_pt.x);
	pass.filter (*cloud_filtered_x);
	    
	pass.setInputCloud (cloud_filtered_x);
	pass.setFilterFieldName ("y");
	pass.setFilterLimits (min_pt.y,max_pt.y);
	pass.filter (*cloud_filtered_xy);
	    
	pass.setInputCloud (cloud_filtered_xy);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (min_pt.z,max_pt.z);
	pass.filter (*cloud_filtered_xyz);

	if(!cloud_filtered_xyz->points.empty()){
	  cv::Point2i p_min(10000,10000);
	  cv::Point2i p_max(-10000,-10000);
	  
	  for(int i=0; i<cloud_filtered_xyz->points.size(); i++){
	    Eigen::Vector3f camera_point = depth_camera_transform.inverse()*
	      Eigen::Vector3f(cloud_filtered_xyz->points[i].x,
			      cloud_filtered_xyz->points[i].y,
			      cloud_filtered_xyz->points[i].z);
	    Eigen::Vector3f image_point = _K*camera_point;

	    const float& z=image_point.z();
	    image_point.head<2>()/=z;
	    int r = image_point.x();
	    int c = image_point.y();

	    if(r < p_min.x)
	      p_min.x = r;
	    if(r > p_max.x)
	      p_max.x = r;

	    if(c < p_min.y)
	      p_min.y = c;
	    if(c > p_max.y)
	      p_max.y = c;

	    cv::circle(image,
	    	       cv::Point2i(r,c),
	    	       1,
	    	       cv::Scalar(255,0,0));
	  }
	  cv::rectangle(image,
			p_min,
			p_max,
			cv::Scalar(0,0,255));
	}

	*objects_cloud_msg += *cloud_filtered_xyz;

      }

      sensor_msgs::ImagePtr label_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

      _label_image_pub.publish(label_image_msg);


    }

  }

private:
  ros::NodeHandle _nh;
  string _robotname;

  ros::Subscriber _camera_info_sub;
  Eigen::Matrix3f _K;
  bool _got_info;

  tf::TransformListener _listener;
  
  message_filters::Subscriber<LogicalImage> _logical_image_sub;
  message_filters::Subscriber<PointCloud> _depth_cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
  typedef sync_policies::ApproximateTime<LogicalImage,PointCloud,sensor_msgs::Image> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  image_transport::ImageTransport _it;
  image_transport::Publisher _label_image_pub;
  
  int _seq;

  
  Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p){
    Eigen::Isometry3f iso;
    iso.translation().x()=p.getOrigin().x();
    iso.translation().y()=p.getOrigin().y();
    iso.translation().z()=p.getOrigin().z();
    Eigen::Quaternionf q;
    tf::Quaternion tq = p.getRotation();
    q.x()= tq.x();
    q.y()= tq.y();
    q.z()= tq.z();
    q.w()= tq.w();
    iso.linear()=q.toRotationMatrix();
    return iso;
  }

  tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T){
    Eigen::Quaternionf q(T.linear());
    Eigen::Vector3f t=T.translation();
    tf::Transform tft;
    tft.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    tft.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    return tft;
  }
  
  const float low=-std::numeric_limits<int>::max();
  const float up=std::numeric_limits<int>::max();

};

int main (int argc, char** argv){
  
  ros::init(argc, argv, "image_labeler");

  ImageLabeler image_labeler;

  ros::spin();
  return 0;
}
