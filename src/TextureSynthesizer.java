import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

public class TextureSynthesizer {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private Mat inputTexture;
    private int resultWidth = 500;
    private int resultHeight = 500;
    private Mat outputTexture;
    private int blockSide = 32;
    private int mergeLength = blockSide / 6;
    private double tolerance = 0.1;
    private List<Mat> blocks = new ArrayList<>();
    private Random random = new Random();

    public static void main(String[] args) {
        TextureSynthesizer ts = new TextureSynthesizer();
        Imgcodecs.imwrite("output.jpg", ts.synthesize());
    }

    private void splitInputImage() {
        int width = inputTexture.width(), height = inputTexture.height();
        int xStart = 0, yStart = 0;
        for (int x = 0; xStart < width; x++) {
            int blockWidth = Integer.min(blockSide, width - x*blockSide);
            if (blockWidth < mergeLength)
                break;

            for (int y = 0; yStart < width; y++) {
                int blockHeight = Integer.min(blockSide, height - y*blockSide);
                if (blockHeight < mergeLength)
                    break;

                Rect rectangle = new Rect(xStart, yStart, blockWidth, blockHeight);
                Mat[] block = new Mat[4];
                block[0] = new Mat(inputTexture, rectangle);
                block[1] = new Mat(); block[2] = new Mat(); block[3] = new Mat();
                Core.rotate(block[0], block[1], Core.ROTATE_90_COUNTERCLOCKWISE);
                Core.rotate(block[1], block[2], Core.ROTATE_90_COUNTERCLOCKWISE);
                Core.rotate(block[2], block[3], Core.ROTATE_90_COUNTERCLOCKWISE);
                blocks.addAll(Arrays.asList(block));

                yStart +=blockSide;
            }
            yStart = 0;
            xStart += blockSide;
        }
    }

    private Mat synthesize() {
        inputTexture = Imgcodecs.imread("textures/cells.jpg");
        outputTexture = new Mat(resultHeight, resultWidth, inputTexture.type());
        splitInputImage();

        int x = 0, y = 0;
        int[] newXY = new int[] {0, 0};
        while (newXY[0] != resultWidth || newXY[1] != resultHeight) {
            Mat bestBlock = getBestMatch(x, y);
            newXY = placeBlock(x, y, bestBlock);
            x = newXY[0];
            if (x == resultWidth) {
                y = newXY[1];
                x = 0;
            }
        }

        return outputTexture;
    }

    // TODO: maybe do caching.
    private Mat getBestMatch(int x, int y) {
        class DiffPair implements Comparable<DiffPair> {
            double diffNorm;
            Mat raster;
            public DiffPair(double norm, Mat r) {
                diffNorm = norm;
                raster = r;
            }
            @Override
            public int compareTo(DiffPair o) {
                return Double.compare(diffNorm, o.diffNorm);
            }
        }
        // calculate error for all blocks and place them in sorted order
        // pick random block within error tolerance*bestBlockError

        PriorityQueue<DiffPair> queue = new PriorityQueue<>(blocks.size());
        for (Mat r : blocks) {
            double diff = calcDiff(x, y, r);
            queue.add(new DiffPair(diff, r));
        }

        ArrayList<Mat> pickList = new ArrayList<>();
        double threshold = queue.peek().diffNorm;
        threshold = (int)((1.0f + tolerance) * threshold);
        while (!queue.isEmpty()) {
            DiffPair d = queue.poll();
            if (d.diffNorm <= threshold) {
                pickList.add(d.raster);
            } else {
                break;
            }
        }

        return pickList.get(random.nextInt(pickList.size()));
    }

    private double calcDiff(int imgX, int imgY, Mat block) {
        if (imgY > mergeLength && imgX > mergeLength) {
            block = limitBlock(imgX - mergeLength, imgY - mergeLength, block);
            Rect hImgRect = new Rect(imgX - mergeLength, imgY - mergeLength, block.width(), mergeLength),
                    hBlockRect = new Rect(0, 0, block.width(), mergeLength),
                    vImgRect = new Rect(imgX - mergeLength, imgY - mergeLength, mergeLength, block.height()),
                    vBlockRect = new Rect(0, 0, mergeLength, block.height());
            Mat hImg = new Mat(outputTexture, hImgRect),
                    hBlock = new Mat(block, hBlockRect),
                    vImg = new Mat(outputTexture, vImgRect),
                    vBlock = new Mat(block, vBlockRect);

            double norm = Core.norm(hImg, hBlock, Core.NORM_L2);
            norm += Core.norm(vImg, vBlock, Core.NORM_L2);

            return norm;
        } else if (imgY > mergeLength) {
            block = limitBlock(imgX, imgY - mergeLength, block);
            Rect hImgRect = new Rect(imgX, imgY - mergeLength, block.width(), mergeLength),
                    hBlockRect = new Rect(0, 0, block.width(), mergeLength);
            Mat hImg = new Mat(outputTexture, hImgRect),
                    hBlock = new Mat(block, hBlockRect);
            return Core.norm(hImg, hBlock, Core.NORM_L2);
        } else if (imgX > mergeLength) {
            block = limitBlock(imgX - mergeLength, imgY, block);
            Rect vImgRect = new Rect(imgX - mergeLength, imgY, mergeLength, block.height()),
                    vBlockRect = new Rect(0, 0, mergeLength, block.height());
            Mat vImg = new Mat(outputTexture, vImgRect),
                    vBlock = new Mat(block,vBlockRect);

            return Core.norm(vImg, vBlock, Core.NORM_L2);
        }

        return 0.0;
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
        block = limitBlock(x, y, block);

        Rect replacedLoc = new Rect(x, y, block.width(), block.height());
        Mat replacedROI = new Mat(outputTexture, replacedLoc);
        // TODO: When replacing merging loc use mask!
        block.copyTo(replacedROI);

        return new int[] {x + block.width(), y + block.height()};
    }

}
