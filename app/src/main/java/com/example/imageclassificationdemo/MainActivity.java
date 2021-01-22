package com.example.imageclassificationdemo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImageBuffer;
    private  int imageSizeX;
    private  int imageSizeY;
    private  TensorBuffer outputProbabilityBuffer;
    private  TensorProcessor probabilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    public static final int PICK_USER_PROFILE_IMAGE = 1000;
    private static final int CAMERA_REQUEST = 1888;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;


    private Bitmap bitmap;
    private List<String> labels;
    ImageView imageView;
    Uri imageuri;
    Button buclassify;
    TextView classitext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView=(ImageView)findViewById(R.id.image);
        buclassify=(Button)findViewById(R.id.classify);
        classitext=(TextView)findViewById(R.id.classifytext);

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
                }
                else
                {
                    Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
                }

            }
        });

        try{
            tflite=new Interpreter(loadmodelfile(this));
        }catch (Exception e) {
            e.printStackTrace();
        }

        buclassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                int imageTensorIndex = 0;
                int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
                imageSizeY = imageShape[1];
                imageSizeX = imageShape[2];
                DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                int probabilityTensorIndex = 0;
                int[] probabilityShape =
                        tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
                DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

                inputImageBuffer = new TensorImage(imageDataType);
                outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
                probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

                inputImageBuffer = loadImage(bitmap);

                tflite.run(inputImageBuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
                showresult();
            }
        });



    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor=activity.getAssets().openFd("modelfff.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private void showresult(){

        try{
            labels = FileUtil.loadLabels(this,"labelsf.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap =(Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            if (entry.getValue()==maxValueInMap) {
                classitext.setText(entry.getKey());
            }
        }
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE)
        {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
            else
            {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
    }


//    @Override
//        protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//            super.onActivityResult(requestCode, resultCode, data);
//
//            if (resultCode == RESULT_OK) {
//                if (requestCode == Constants.PICK_USER_PROFILE_IMAGE) {
//                    if (resultCode == RESULT_OK) {
//                        Bitmap bmp = ImagePicker.getImageFromResult(this, resultCode, data);
//                        //your compressed rotated bitmap here
//
//                    }
//                }
//            }
//        }
}

