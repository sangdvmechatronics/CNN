+ (CameraPose)findCameraPose: (NSArray<NSValue *> *) objectPoints imagePoints: (NSArray<NSValue *> *) imagePoints size: (CGSize) size {

    vector<Point3f> cvObjectPoints = [self convertObjectPoints:objectPoints];
    vector<Point2f> cvImagePoints = [self convertImagePoints:imagePoints withSize: size];

    std::cout << "object points: " << cvObjectPoints << std::endl;
    std::cout << "image points: " << cvImagePoints << std::endl;

    cv::Mat distCoeffs(4,1,cv::DataType<double>::type, 0.0);
    cv::Mat rvec(3,1,cv::DataType<double>::type);
    cv::Mat tvec(3,1,cv::DataType<double>::type);
    cv::Mat cameraMatrix = [self intrinsicMatrixWithImageSize: size];

    cv::solvePnP(cvObjectPoints, cvImagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    std::cout << "rvec: " << rvec << std::endl;
    std::cout << "tvec: " << tvec << std::endl;

    std::vector<cv::Point2f> projectedPoints;
    cvObjectPoints.push_back(Point3f(0.0, 0.0, 0.0));
    cv::projectPoints(cvObjectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    for(unsigned int i = 0; i < projectedPoints.size(); ++i) {
        std::cout << "Image point: " << cvImagePoints[i] << " Projected to " << projectedPoints[i] << std::endl;
    }


    cv::Mat RotX(3, 3, cv::DataType<double>::type);
    cv::setIdentity(RotX);
    RotX.at<double>(4) = -1; //cos(180) = -1
    RotX.at<double>(8) = -1;

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    R = R.t();  // rotation of inverse
    Mat rvecConverted;
    Rodrigues(R, rvecConverted); //
    std::cout << "rvec in world coords:\n" << rvecConverted << std::endl;
    rvecConverted = RotX * rvecConverted;
    std::cout << "rvec scenekit :\n" << rvecConverted << std::endl;

    Mat tvecConverted = -R * tvec;
    std::cout << "tvec in world coords:\n" << tvecConverted << std::endl;
    tvecConverted = RotX * tvecConverted;
    std::cout << "tvec scenekit :\n" << tvecConverted << std::endl;

    SCNVector4 rotationVector = SCNVector4Make(rvecConverted.at<double>(0), rvecConverted.at<double>(1), rvecConverted.at<double>(2), norm(rvecConverted));
    SCNVector3 translationVector = SCNVector3Make(tvecConverted.at<double>(0), tvecConverted.at<double>(1), tvecConverted.at<double>(2));

    return CameraPose{rotationVector, translationVector};
}