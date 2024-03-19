package com.ece420.lab7;

import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.Core.getTickCount;
import static org.opencv.core.Core.getTickFrequency;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8UC3;

import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.CompoundButton;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerKCF;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner; // file reader ipehlivan
import java.io.File;
import java.io.FileNotFoundException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    // UI Variables
    private Button controlButton;

    private Button methodButton;  // Used to switch between eigenfaces & fisherfaces
    private SeekBar colorSeekbar;
    private SeekBar widthSeekbar;
    private SeekBar heightSeekbar;
    private TextView widthTextview;
    private TextView heightTextview;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;

    // KCF Tracker variables
    private TrackerKCF myTacker;
    private Rect2d myROI = new Rect2d(0,0,0,0);
    private int myROIWidth = 70;
    private int myROIHeight = 70;
    private Scalar myROIColor = new Scalar(0,0,0);
    private int tracking_flag = -1;

    // Variables used for Eigenface algorithm
    private int algorithm_button; // Flag to tell us status of methodButton (1 = Eigenfaces, 0 Fisherfaces, -1 = Not chosen yet)
    private int eigen_vecs_to_skip = 2; // KNN, how many eigenvectors to skip.
    private String name_prediction = "Align ROI with face!";
    private int numPicsPerPerson = 3;
    private int totalPicsofUs = numPicsPerPerson * 3;
    private int training_shape_x = 195;
    private int training_shape_y = 231;
    private int eigenmatrix_width = training_shape_x * training_shape_y;
    private int num_faces = 14; // Number of Eigenfaces generated
    private int[] labels = new int[numPicsPerPerson*3];
    private Mat eigenmatrix; // Declare (num_faces)-by-(eigenmatrix_width) matrix to store the eigenvectors
    private Mat eigen_mean; // Declare 1-by-(eigenmatrix_width) matrix to store the mean values of eigenvectors
    private Mat flattened_gray; // Declare 1-by-(eigenmatrix_width) matrix to store the result of (resized ROI - eigen_mean)
    private Mat flattened_gray_trans; // Declare Transposed version of flattened_gray, (eigenmatrix_width)-by-1 matrix
    private Mat final_data; // Declare Output of matrix multiplication of eigenmatrix & flattened_gray_trans
    private Mat final_data_trans; // Declare Transposed version of final_data, 1-by-(num_faces) matrix
    private Mat ACR; // Declare (totalPicsofUs)-by-(num_faces) matrix to store the ACR data
    List list_eigen_ACR;
    List list_eigen_MeanAndFaces;
    List list_fisher_ACR;
    List list_fisher_MeanAndFaces;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        // Load .csv files from memory, and make ready to be accessed.
        InputStream ACR_eigen_is = getResources().openRawResource(R.raw.acr_eigen); // Input stream for acr_eigen.csv
        CSVFile csv_file_ACR_eigen = new CSVFile(ACR_eigen_is);
        list_eigen_ACR = csv_file_ACR_eigen.read();

        InputStream MeanAndFaces_eigen_is = getResources().openRawResource(R.raw.meanandfaces_eigen); // Input stream for meanandfaces_eigen.csv
        CSVFile csv_file_MeanAndFaces_eigen = new CSVFile(MeanAndFaces_eigen_is);
        list_eigen_MeanAndFaces = csv_file_MeanAndFaces_eigen.read();

        InputStream ACR_fisher_is = getResources().openRawResource(R.raw.acr_fisher); // Input stream for acr_fisher.csv
        CSVFile csv_file_ACR_fisher = new CSVFile(ACR_fisher_is);
        list_fisher_ACR = csv_file_ACR_fisher.read();

        InputStream MeanAndFaces_fisher_is = getResources().openRawResource(R.raw.meanandfaces_fisher); // Input stream for meanandfaces_fisher.csv
        CSVFile csv_file_MeanAndFaces_fisher = new CSVFile(MeanAndFaces_fisher_is);
        list_fisher_MeanAndFaces = csv_file_MeanAndFaces_fisher.read();

        // Setup color seek bar
        colorSeekbar = (SeekBar) findViewById(R.id.colorSeekBar);
        colorSeekbar.setProgress(50);
        setColor(50);
        colorSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                setColor(progress);
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        widthTextview = (TextView) findViewById(R.id.widthTextView);
        widthSeekbar = (SeekBar) findViewById(R.id.widthSeekBar);
        widthSeekbar.setProgress(myROIWidth - 20);
        widthSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIWidth = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        heightTextview = (TextView) findViewById(R.id.heightTextView);
        heightSeekbar = (SeekBar) findViewById(R.id.heightSeekBar);
        heightSeekbar.setProgress(myROIHeight - 20);
        heightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIHeight = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });




        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (tracking_flag == -1) {
                    // Modify UI
                    controlButton.setText("STOP TRACKING!");
                    widthTextview.setVisibility(View.INVISIBLE);
                    widthSeekbar.setVisibility(View.INVISIBLE);
                    heightTextview.setVisibility(View.INVISIBLE);
                    heightSeekbar.setVisibility(View.INVISIBLE);
                    // Modify tracking flag
                    tracking_flag = 0;
//                    if(method == )
                }
                else if(tracking_flag == 1){
                    // Modify UI
                    controlButton.setText("START TRACKING!");
                    widthTextview.setVisibility(View.VISIBLE);
                    widthSeekbar.setVisibility(View.VISIBLE);
                    heightTextview.setVisibility(View.VISIBLE);
                    heightSeekbar.setVisibility(View.VISIBLE);
                    // Tear down myTracker
                    myTacker.clear();
                    // Modify tracking flag
                    tracking_flag = -1;
                    // Set prediction text blank
                    name_prediction = "Align ROI with face!";
                }
            }
        });


        // Setup method switch, this will control if we will be tracking with the eigenface or fisherface tracking algorithm
        Switch sw = (Switch) findViewById(R.id.methodButton);
        sw.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                algorithm_button = -1;
                if (isChecked) { //ON = Eigenfaces
                    // The toggle is enabled
                    algorithm_button = 1;

                    String[] l;
                    for (int i = 1+eigen_vecs_to_skip; i < list_eigen_MeanAndFaces.size(); i++) {
                        l = (String[]) list_eigen_MeanAndFaces.get(i);

                        for (int j = 0; j < l.length; j++) {
                            // Parsing string to double
                            eigenmatrix.put(i,j,Double.parseDouble(l[j]));
                        }
                    }

                    String[] l_1;
                    for (int i = 0; i < list_eigen_ACR.size(); i++) {
                        l_1 = (String[]) list_eigen_ACR.get(i);

                        for (int j = 0; j < l_1.length; j++) {
                            // Parsing string to double
                            ACR.put(i,j,Double.parseDouble(l_1[j]));
                        }
                    }

                    String[] l_2;
                    for (int i = 0; i < 1; i++) {
                        l_2 = (String[]) list_eigen_MeanAndFaces.get(i);

                        for (int j = 0; j < l_2.length; j++) {
                            // Parsing string to double
                            eigen_mean.put(i,j,Double.parseDouble(l_2[j]));
                        }
                    }


                    for (int i = 0; i < 3; i++) {
                        for (int j = i*numPicsPerPerson; j < i*numPicsPerPerson+numPicsPerPerson; j++) {
                            labels[j] = i;
                        }
                    }


                } else { //OFF = Fisherfaces
                    // The toggle is disabled
                    algorithm_button = 0;

                    String[] l;
                    for (int i = 1+eigen_vecs_to_skip; i < list_fisher_MeanAndFaces.size(); i++) {
                        l = (String[]) list_fisher_MeanAndFaces.get(i);

                        for (int j = 0; j < l.length; j++) {
                            // Parsing string to double
                            eigenmatrix.put(i,j,Double.parseDouble(l[j]));
                        }
                    }

                    String[] l_1;
                    for (int i = 0; i < list_fisher_ACR.size(); i++) {
                        l_1 = (String[]) list_fisher_ACR.get(i);

                        for (int j = 0; j < l_1.length; j++) {
                            // Parsing string to double
                            ACR.put(i,j,Double.parseDouble(l_1[j]));
                        }
                    }

                    String[] l_2;
                    for (int i = 0; i < 1; i++) {
                        l_2 = (String[]) list_fisher_MeanAndFaces.get(i);

                        for (int j = 0; j < l_2.length; j++) {
                            // Parsing string to double
                            eigen_mean.put(i,j,Double.parseDouble(l_2[j]));
                        }
                    }


                    for (int i = 0; i < 3; i++) {
                        for (int j = i*numPicsPerPerson; j < i*numPicsPerPerson+numPicsPerPerson; j++) {
                            labels[j] = i;
                        }
                    }

                    //name_prediction = "No CLUE!";
                }
            }
        });

        //OpenCVLoader.initDebug();

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(1);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        // mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    eigenmatrix = new Mat(num_faces, eigenmatrix_width, CvType.CV_64F); // Initialize (num_faces)-by-(eigenmatrix_width) matrix to store the eigenvectors
                    eigen_mean = new Mat(1, eigenmatrix_width, CvType.CV_64F); // Initialize 1-by-(eigenmatrix_width) matrix to store the mean values of eigenvectors
                    flattened_gray = new Mat(1, eigenmatrix_width, CvType.CV_64F); // Initialize 1-by-(eigenmatrix_width) matrix to store the result of (resized ROI - eigen_mean)
                    flattened_gray_trans = new Mat(eigenmatrix_width, 1, CvType.CV_64F); // Initialize Transposed version of flattened_gray, (eigenmatrix_width)-by-1 matrix
                    final_data = new Mat(num_faces, 1, CvType.CV_64F); // Initialize Output of matrix multiplication of eigenmatrix & flattened_gray_trans
                    final_data_trans = new Mat(1, num_faces, CvType.CV_64F); // Initialize Transposed version of final_data, 1-by-(num_faces) matrix
                    ACR = new Mat(totalPicsofUs, num_faces, CvType.CV_64F); // // Initialize (totalPicsofUs)-by-(num_faces) matrix to store the ACR data

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    // https://javapapers.com/android/android-read-csv-file/
    // Following class is an utility to read CSV file
    public class CSVFile {
        InputStream inputStream;

        public CSVFile(InputStream inputStream){
            this.inputStream = inputStream;
        }

        public List read(){
            List resultList = new ArrayList();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            try {
                String csvLine;
                while ((csvLine = reader.readLine()) != null) {
                    String[] row = csvLine.split(",");
                    resultList.add(row);
                }
            }
            catch (IOException ex) {
                throw new RuntimeException("Error in reading CSV file: "+ex);
            }
            finally {
                try {
                    inputStream.close();
                }
                catch (IOException e) {
                    throw new RuntimeException("Error while closing input stream: "+e);
                }
            }
            return resultList;
        }
    }


    // Helper Function to map single integer to color scalar
    // https://www.particleincell.com/2014/colormap/
    public void setColor(int value) {
        double a=(1-(double)value/100)/0.2;
        int X=(int)Math.floor(a);
        int Y=(int)Math.floor(255*(a-X));
        double newColor[] = {0,0,0};
        switch(X)
        {
            case 0:
                // r=255;g=Y;b=0;
                newColor[0] = 255;
                newColor[1] = Y;
                break;
            case 1:
                // r=255-Y;g=255;b=0
                newColor[0] = 255-Y;
                newColor[1] = 255;
                break;
            case 2:
                // r=0;g=255;b=Y
                newColor[1] = 255;
                newColor[2] = Y;
                break;
            case 3:
                // r=0;g=255-Y;b=255
                newColor[1] = 255-Y;
                newColor[2] = 255;
                break;
            case 4:
                // r=Y;g=0;b=255
                newColor[0] = Y;
                newColor[2] = 255;
                break;
            case 5:
                // r=255;g=0;b=255
                newColor[0] = 255;
                newColor[2] = 255;
                break;
        }
        myROIColor.set(newColor);
        return;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CV_8UC1);
        myWidth = width;
        myHeight = height;
        myROI = new Rect2d(myWidth / 2 - myROIWidth / 2,
                            myHeight / 2 - myROIHeight / 2,
                            myROIWidth,
                            myROIHeight);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Timer
        long start = getTickCount();
        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        Mat mRgba_flipped = inputFrame.rgba();
        Mat mGray_flipped = inputFrame.gray();

        // Invert put camera inputs:
        Core.flip(mRgba, mRgba_flipped, 1);
        Core.flip(mGray, mGray_flipped, 1);

//        Mat mGray_cropped(myWidth, myHeight, )
//        mGray_cropped = inputFrame.gray();

        // Action based on tracking flag
        if(tracking_flag == -1){
            // Update myROI to keep the window to the center
            myROI.x = myWidth / 2 - myROIWidth / 2;
            myROI.y = myHeight / 2 - myROIHeight / 2;
            myROI.width = myROIWidth;
            myROI.height = myROIHeight;
        }
        else if(tracking_flag == 0){
            // Initialize KCF Tracker and Start Tracking
            // 1. Create a KCF Tracker
            // 2. Initialize KCF Tracker with grayscale image and ROI
            // 3. Modify tracking flag to start tracking

            myTacker = TrackerKCF.create();
            myTacker.init(mGray_flipped, myROI);
            tracking_flag = 1;

            // Crop ROI from image
            Mat mCrop = mGray.submat((int) (myROI.y),(int) (myROI.y+myROIHeight),(int) (myROI.x), (int) (myROI.x+myROIWidth));

            // Resize the image by training_shape_x*training_shape_y size of our training data
            Mat mResized = new Mat(training_shape_x, training_shape_y, CvType.CV_64F);
            Imgproc.resize(mCrop,mResized,mResized.size(), 0,0, Imgproc.INTER_AREA);

            // Vectorize the image
            mResized = mResized.reshape(1,1);
            mResized.convertTo(mResized,CvType.CV_64F);

            // Subtract the mean
            Core.subtract(mResized, eigen_mean, flattened_gray);

            // Transpose output for matrix multiplication
            Core.transpose(flattened_gray, flattened_gray_trans);

            // Matrix Multiplication
            Core.gemm(eigenmatrix, flattened_gray_trans, 1, new Mat(), 0.0, final_data, 0);

            // Transpose back to row vector
            Core.transpose(final_data, final_data_trans);

            // Run Classifiers (see how close final_data is to one of of rows in ACR, and choose that label)
            Mat distanceMat = new Mat(ACR.rows(), ACR.cols(), ACR.type()); // Mat to compare distance of current ROI from eigenfaces
            Mat repeatedRow = new Mat(final_data_trans.rows(), ACR.cols(), ACR.type()); // Temp to hold the eigenface we are currently comparing against
            double[] euc_dist = new double[ACR.rows()]; // array to hold the Euclidean distance between current ROI & eigenfaces

            for (int row = 0; row < ACR.rows(); row++) {
                // fill repeated row with ACR data
                Core.repeat(ACR.row(row), 1, 1, repeatedRow);

                // calculate the distance between each row of the new Mat object and the 1D Mat object
                Core.absdiff(repeatedRow, final_data_trans, distanceMat.row(row));
                euc_dist[row] = Core.norm(distanceMat.row(row), Core.NORM_L2); //-1
            }

            // find the row index with the smallest distance
            int minIndex = 0;
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < ACR.rows(); i++) {
                if (euc_dist[i] < minDistance) {
                    minIndex = i;
                    minDistance = euc_dist[i];
                }
            }

            if (labels[minIndex] == 0){
                name_prediction = "Prediction = Aahan";
            }
            else if (labels[minIndex] == 1){
                name_prediction = "Prediction = Chaz";
            }
            else if (labels[minIndex] == 2){
                name_prediction = "Prediction = Rutvik";
            }

        }
        else{
            // Update tracking result is succeed
            // If failed, print text "Tracking failure occurred!" at top left corner of the frame
            // Calculate and display "FPS@fps_value" at top right corner of the frame

            // We recommend using the grayscale frame mGray for tracking
            // and only use mRgba for display and preview purposes.

            boolean read_success = myTacker.update(mGray_flipped, myROI);

            if (read_success == false) {
                Imgproc.putText (
                        mRgba_flipped,                          // Matrix obj of the image
                        "Tracking failure occurred",          // Text to be added
                        new Point(10, 30),               // point
                        Core.FONT_HERSHEY_SIMPLEX ,      // front face
                        1,                               // front scale
                        new Scalar(255, 0, 0),             // Scalar object for color
                        2                                // Thickness
                );
            }

            double fps_value = getTickFrequency() / (getTickCount() - start);
            //int fps_value = int v;
            //cv2.putText(frame, , (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2);

            Imgproc.putText (
                    mRgba_flipped,                          // Matrix obj of the image
                    "FPS@"+ ((int)fps_value),          // Text to be added
                    new Point(500, 30),               // point
                    Core.FONT_HERSHEY_SIMPLEX ,      // front face
                    1.25,                               // front scale
                    new Scalar(0, 255, 0),             // Scalar object for color
                    2                                // Thickness
            );

        }

        // Draw a rectangle on to the current frame
        Imgproc.rectangle(mRgba_flipped,
                          new Point(myROI.x, myROI.y),
                          new Point(myROI.x + myROI.width, myROI.y + myROI.height),
                          myROIColor,
                4);

        Imgproc.putText (
                mRgba_flipped,                          // Matrix obj of the image
                name_prediction,          // Text to be added
                new Point(myROI.x, myROI.y - 10),               // point
                Core.FONT_HERSHEY_SIMPLEX ,      // front face
                1.25,                               // front scale
                new Scalar(0, 255, 0),             // Scalar object for color
                2                                // Thickness
        );

        if (algorithm_button == 0) {
            Imgproc.putText (
                    mRgba_flipped,                          // Matrix obj of the image
                    "Algorithm = Fisherfaces",          // Text to be added
                    new Point(500, 60),               // point
                    Core.FONT_HERSHEY_SIMPLEX ,      // front face
                    1.25,                               // front scale
                    new Scalar(0, 255, 0),             // Scalar object for color
                    2                                // Thickness
            );
        }
        else if (algorithm_button == 1) {
            Imgproc.putText (
                    mRgba_flipped,                          // Matrix obj of the image
                    "Algorithm = Eigenfaces",          // Text to be added
                    new Point(500, 60),               // point
                    Core.FONT_HERSHEY_SIMPLEX ,      // front face
                    1.25,                               // front scale
                    new Scalar(0, 255, 0),             // Scalar object for color
                    2                                // Thickness
            );
        }
        else {
            Imgproc.putText (
                    mRgba_flipped,                          // Matrix obj of the image
                    "Algorithm not functioning",          // Text to be added
                    new Point(500, 60),               // point
                    Core.FONT_HERSHEY_SIMPLEX ,      // front face
                    1.25,                               // front scale
                    new Scalar(0, 255, 0),             // Scalar object for color
                    2                                // Thickness
            );
        }

        // Returned frame will be displayed on the screen
        return mRgba_flipped;
    }
}