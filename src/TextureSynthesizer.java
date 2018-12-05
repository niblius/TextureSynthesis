import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

public class TextureSynthesizer {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private Mat inputTexture;
    private int resultWidth = 750;
    private int resultHeight = 750;
    private Mat outputTexture;
    static int blockSide = 50;
    private int mergeLength = 15;
    private int stepSize = 1;
    private double tolerance = 0.1;
    private List<Mat> blocks = new ArrayList<>();
    // TODO: random seed
    private Random random = new Random();

    static boolean debug = true;
    private Mat debugMask;
    private Mat debugCutPath;

    public static void main(String[] args) {
        TextureSynthesizer ts = new TextureSynthesizer();
        Imgcodecs.imwrite("output.jpg", ts.synthesize());
        if (debug) {
            Imgcodecs.imwrite("mask.jpg", ts.debugMask);
            Imgcodecs.imwrite("path2.jpg", ts.debugCutPath);
        }
    }

    private void splitInputImage() {
        int width = inputTexture.width(), height = inputTexture.height();
        int xStart = 0, yStart = 0;
        for (int x = 0; xStart < width; x++) {
            int blockWidth = Integer.min(blockSide, width - x*stepSize);
            if (blockWidth < mergeLength)
                break;

            for (int y = 0; yStart < width; y++) {
                int blockHeight = Integer.min(blockSide, height - y*stepSize);
                if (blockHeight < mergeLength)
                    break;

                Rect rectangle = new Rect(xStart, yStart, blockWidth, blockHeight);
                Mat block = new Mat(inputTexture, rectangle);

                if (block.width() == blockSide && block.height() == blockSide)
                    blocks.add(block);

                yStart += stepSize;
            }
            yStart = 0;
            xStart += stepSize;
        }
    }

    private void drawDebugBlocks() {
        Mat blocksImage = new Mat(blocks.get(0).height(), blocks.size()*(blockSide+3), inputTexture.type());
        for (int i = 0; i < blocks.size(); i++) {
            Rect r = new Rect(i*(blockSide+3), 0, blockSide, blockSide);
            blocks.get(i).copyTo(blocksImage.submat(r));
        }
        Imgcodecs.imwrite("blocks.jpg", blocksImage);
    }

    private Mat synthesize() {
        inputTexture = Imgcodecs.imread("textures/apples.jpg");
        outputTexture = new Mat(resultHeight, resultWidth, inputTexture.type());
        splitInputImage();
        if (debug) {
            debugMask = new Mat(resultHeight, resultWidth, inputTexture.type());
            debugCutPath = new Mat(resultHeight, resultWidth, inputTexture.type());
            drawDebugBlocks();
        }

        int x = 0, y = 0;
        int[] newXY = new int[] {0, 0};
        while (newXY[0] != resultWidth || newXY[1] != resultHeight) {
            Mat bestBlock = getBestMatch(x, y);
            newXY = placeBlock(x, y, bestBlock);
            x = newXY[0];
            if (x == resultWidth) {
                y = newXY[1];
                x = 0;
                System.out.println("" + (double)y*100/resultHeight + "%");
            }
        }

        return outputTexture;
    }

    // TODO: maybe do caching.
    private Mat getBestMatch(int x, int y) {
        class DiffPair implements Comparable<DiffPair> {
            private double diffNorm;
            private Mat block;
            private DiffPair(double norm, Mat r) {
                diffNorm = norm;
                block = r;
            }
            @Override
            public int compareTo(DiffPair o) {
                return Double.compare(diffNorm, o.diffNorm);
            }
        }

        PriorityQueue<DiffPair> queue = new PriorityQueue<>(blocks.size());
        for (Mat r : blocks) {
            double diff = calcDiff(x, y, r);
            queue.add(new DiffPair(diff, r));
        }

        ArrayList<Mat> pickList = new ArrayList<>();
        double threshold = queue.peek().diffNorm;
        threshold = (1.0f + tolerance) * threshold;
        while (!queue.isEmpty()) {
            DiffPair d = queue.poll();
            if (d.diffNorm <= threshold) {
                pickList.add(d.block);
            } else {
                break;
            }
        }

        return pickList.get(random.nextInt(pickList.size()));
    }


    private double calcDiff(int imgX, int imgY, Mat block) {
        double norm;
        if (imgY > mergeLength && imgX > mergeLength) {
            block = limitBlock(imgX - mergeLength, imgY - mergeLength, block);
            Rect hImgRect = new Rect(imgX - mergeLength, imgY - mergeLength, block.width(), mergeLength),
                    hBlockRect = new Rect(0, 0, block.width(), mergeLength),
                    vImgRect = new Rect(imgX - mergeLength, imgY, mergeLength, block.height() - mergeLength),
                    vBlockRect = new Rect(0, mergeLength, mergeLength, block.height() - mergeLength);
            Mat hImg = new Mat(outputTexture, hImgRect),
                    hBlock = new Mat(block, hBlockRect),
                    vImg = new Mat(outputTexture, vImgRect),
                    vBlock = new Mat(block, vBlockRect);

            norm = Core.norm(hImg, hBlock, Core.NORM_L2SQR);
            norm += Core.norm(vImg, vBlock, Core.NORM_L2SQR);
        } else if (imgY > mergeLength) {
            block = limitBlock(imgX, imgY - mergeLength, block);
            Rect hImgRect = new Rect(imgX, imgY - mergeLength, block.width(), mergeLength),
                    hBlockRect = new Rect(0, 0, block.width(), mergeLength);
            Mat hImg = new Mat(outputTexture, hImgRect),
                    hBlock = new Mat(block, hBlockRect);
            norm = Core.norm(hImg, hBlock, Core.NORM_L2SQR);
        } else if (imgX > mergeLength) {
            block = limitBlock(imgX - mergeLength, imgY, block);
            Rect vImgRect = new Rect(imgX - mergeLength, imgY, mergeLength, block.height()),
                    vBlockRect = new Rect(0, 0, mergeLength, block.height());
            Mat vImg = new Mat(outputTexture, vImgRect),
                    vBlock = new Mat(block,vBlockRect);

            norm = Core.norm(vImg, vBlock, Core.NORM_L2SQR);
        } else {
            norm = 1.0;
        }

        return norm;
    }

    private Mat limitBlock(int x, int y, Mat block) {
        if (resultWidth - x < block.width() || resultHeight - y < block.height()) {
            int newBlockWidth = Integer.min(resultWidth - x, block.width());
            int newBlockHeight = Integer.min(resultHeight - y, block.height());
            Rect rectangle = new Rect(0, 0, newBlockWidth, newBlockHeight);
            return block.submat(rectangle);
        }
        return block;
    }

    private int[] placeBlock(int x, int y, Mat block) {
        int placeX = x, placeY = y;
        if (x > mergeLength)
            placeX -= mergeLength;
        if (y > mergeLength)
            placeY -= mergeLength;

        block = limitBlock(placeX, placeY, block);

        Rect replacedLoc = new Rect(placeX, placeY, block.width(), block.height());
        Mat replacedROI = new Mat(outputTexture, replacedLoc);
        Mat mask = calculateMergeMask(x, y, block);
        block.copyTo(replacedROI, mask);
        if (debug) {
            Mat debugMaskROI = new Mat(debugMask, replacedLoc);
            mask.copyTo(debugMaskROI);
            Mat debugCutPathROI = new Mat(debugCutPath, replacedLoc);
            if (Graph.debugPath != null)
                Graph.debugPath.copyTo(debugCutPathROI);
            else
                debugCutPathROI.setTo(Graph.plusScalar);
        }

        return new int[] {placeX + block.width(), placeY + block.height()};
    }

    private Mat calculateMergeMask(int imgX, int imgY, Mat block) {
        int startX = imgX, starY = imgY, restrictX = 0, restrictY = 0;
        if (imgX > mergeLength && imgY > mergeLength) {
            startX -= mergeLength;
            starY -= mergeLength;
            restrictX = mergeLength;
            restrictY = mergeLength;
        } else if (imgX > mergeLength) {
            startX -= mergeLength;
            restrictX = mergeLength;
        } else if (imgY > mergeLength) {
            starY -= mergeLength;
            restrictY = mergeLength;
        } else {
            return Mat.ones(block.size(), block.type()).setTo(Graph.plusScalar);
        }

        Rect imgROI = new Rect(startX, starY, block.width(), block.height());
        Mat img = new Mat(outputTexture, imgROI);
        Graph graph = new Graph(img, block, restrictX, restrictY);
        return graph.buildMask();
    }

}
