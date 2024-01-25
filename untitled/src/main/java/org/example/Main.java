package org.example;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
    private static final int borderSize = 10;
    private static final double SCALE = 120.0/50.0; // 120 пикселей = 1 микрометр

    public static void main(String[] args) {
        // Загрузка библиотеки OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Путь к изображению
        String imagePath = "clearImg.jpg";
        String imagePathContours = "img.png";

        // Загрузка изображения
        Mat image = Imgcodecs.imread(imagePath);
        Mat imageContours = Imgcodecs.imread(imagePathContours);

        // Поиск контуров
        List<MatOfPoint> contours = findContours(image);

        // Вывод количества точек
        int pointCount = contours.size();
        System.out.println("Количество точек: " + pointCount);

        // Вычисление и вывод площади
        double area = calculateArea(contours);
        System.out.println("Площадь: " + area + "µm");

        // Отрисовка контуров на изображении
        drawPoints(imageContours, contours);

        // Вывод в json файл данных
        saveDataToJson(pointCount, area);
    }

    private static List<MatOfPoint> findContours(Mat image) {

        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);
        Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 3, 1);
        Imgproc.medianBlur(gray, gray, 3);
        gray = addBorderToImage(gray);


        // Сохранение изображения после пороговой обработки
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea = -1;
        int indexOfLargestContour = -1;

        for (int i = 0; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                indexOfLargestContour = i;
            }
        }
        List<Double> d = new ArrayList<>();
        for (MatOfPoint m : contours) {
            d.add(Imgproc.contourArea(m));
        }
        Collections.sort(d);
        Collections.reverse(d);
        contours.remove(indexOfLargestContour);

        return contours;
    }

    private static void saveDataToJson(int pointCount, double area) {
        JsonObject jsonObject = new JsonObject();
        jsonObject.addProperty("Количество точек", pointCount);
        jsonObject.addProperty("Площадь", area + " µm");

        Gson gson = new Gson();
        String jsonData = gson.toJson(jsonObject);
        try (FileWriter writer = new FileWriter("result.json")) {
            writer.write(jsonData);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Mat removeBorderFromImage(Mat image) {
        int newWidth = image.cols() - 2 * borderSize;
        int newHeight = image.rows() - 2 * borderSize;

        // Обрезка изображения для удаления вокругних пикселей
        Rect roi = new Rect(borderSize, borderSize, newWidth, newHeight);

        return new Mat(image, roi).clone();
    }

    private static Mat addBorderToImage(Mat image) {
        int newWidth = image.cols() + 2 * borderSize;
        int newHeight = image.rows() + 2 * borderSize;

        // Создание нового изображения с вокругними пикселямти
        Mat imageWithBorder = new Mat(newHeight, newWidth, image.type(), new Scalar(255, 255, 255));

        // Копирование исходного изображения в центр нового изображения
        Rect roi = new Rect(borderSize, borderSize, image.cols(), image.rows());
        image.copyTo(new Mat(imageWithBorder, roi));

        return imageWithBorder;
    }

    private static double calculateArea(List<MatOfPoint> contours) {
        double area = 0;
        for (MatOfPoint contour : contours) {
            area += Imgproc.contourArea(contour);
        }
        double um = area / SCALE;
        area = area / (um * um);
        return area;
    }


    private static void drawPoints(Mat image, List<MatOfPoint> contours) {

        Mat contoursImage = addBorderToImage(image.clone());
        Imgproc.drawContours(contoursImage, contours, -1, new Scalar(0, 255, 0), 1);
        contoursImage = removeBorderFromImage(contoursImage);
        Imgcodecs.imwrite("contours_image_original.jpg", contoursImage);
    }

}