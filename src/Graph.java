import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Stack;

public class Graph {
    private Mat img1;
    private Mat img2;
    private byte[] buff1;
    private byte[] buff2;
    private int channels;
    private int width;
    private int height;
    private int restrictX;
    private int restrictY;
    private int mergeLength;
    private Point[][] field;
    private Mat mask;
    private byte[] maskBuff;

    public static Mat debugPath;
    private static byte[] debugPathBuff;

    private enum PointType {
        START, END, ORDINARY, PATH
    }
    private class Point implements Comparable<Point> {
        boolean visited = false;
        int distance;
        int x;
        int y;
        PointType type = PointType.ORDINARY;
        Point prev;

        Point(int x, int y) {
            this.x = x;
            this.y = y;
        }

        List<Point> getNeighbors() {
            ArrayList<Point> neighbors = new ArrayList<>(4);
            if (x-1 >= 0) {
                neighbors.add(field[x-1][y]);
            }
            if (y-1 >= 0) {
                neighbors.add(field[x][y-1]);
            }

            if ((x < restrictX && y+1 < height) || (x > restrictX && y+1 < restrictY)){
                neighbors.add(field[x][y+1]);
            }

            if ((y < restrictY && x+1 < width) || (y > restrictY && x+1 < restrictX)) {
                neighbors.add(field[x+1][y]);
            }

            return neighbors;
        }

        @Override
        public int compareTo(Point o) {
            return Integer.compare(distance, o.distance);
        }
    }

    public Graph(Mat img1, Mat img2, int restrictX, int restrictY, int mergeLength) {
        this.mergeLength = mergeLength;
        this.restrictX = restrictX;
        this.restrictY = restrictY;
        this.img1 = img1;
        this.img2 = img2;
        width = img1.width();
        height = img1.height();
        channels = img1.channels();
        buff1 = new byte[(int)img1.total() * channels];
        buff2 = new byte[(int)img1.total() * channels];
        img1.get(0, 0, buff1);
        img2.get(0, 0, buff2);
    }

    private int getDifference(int x, int y) {
        int norm = 0;
        for (int i = 0; i < channels; i++) {
            double diff = buff1[(y*width + x)*channels + i] - buff2[(y*width + x)*channels + i];
            norm += diff * diff;
        }
        return norm;
    }

    private void pathSearch() {
        // TODO: use array of arrays, because we won't need entire matrix
        // only its small part
        field = new Point[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                field[x][y] = new Point(x, y);
            }
        }

        for (int y = 0; y < mergeLength; y++) {
            field[width-1][y].type = PointType.END;
        }

        PriorityQueue<Point> queue = new PriorityQueue<>();
        for (int x = 0; x < mergeLength; x++) {
            Point p = field[x][height-1];
            p.type = PointType.START;
            p.distance = getDifference(p.x, p.y);
            p.visited = true;
            queue.add(p);
        }

        dijkstra(queue);
        if (TextureSynthesizer.debug) {
            debugPath = Mat.ones(img1.size(), img1.type()).setTo(plusScalar);
            debugPathBuff = new byte[(int) img1.total() * channels];
            debugPath.get(0, 0, debugPathBuff);
        }
        recoverPath();
    }

    private void recoverPath() {
        Point end = null;
        int min = Integer.MAX_VALUE;
        for (int y = 0; y < mergeLength; y++) {
            Point p = field[width-1][y];
            if (p.distance < min) {
                min = p.distance;
                end = p;
            }
        }

        do {
            if (TextureSynthesizer.debug)
                setEmpty(end.x, end.y, debugPathBuff);

            end.type = PointType.PATH;
            end = end.prev;
        } while(end.type != PointType.START);
        end.type = PointType.PATH;

        if (TextureSynthesizer.debug)
            debugPath.put(0, 0, debugPathBuff);
    }

    private void dijkstra(PriorityQueue<Point> queue) {
        while (!queue.isEmpty()) {
            Point p = queue.poll();
            p.visited = true;
            for (Point neighbor : p.getNeighbors()) {
                int d = p.distance + getDifference(neighbor.x, neighbor.y);
                if (!neighbor.visited) {
                    neighbor.distance = p.distance + getDifference(neighbor.x, neighbor.y);
                    neighbor.prev = p;
                    queue.add(neighbor);
                } else if (neighbor.distance > d) {
                    queue.remove(neighbor);
                    neighbor.distance = d;
                    neighbor.prev = p;
                    queue.add(neighbor);
                }
            }
        }
    }

    private void setEmpty(int x, int y, byte[] buff) {
        for (int i = 0; i < channels; i++) {
            buff[(y*width + x)*channels + i] = 0;
        }
    }

    private static final Scalar plusScalar = new Scalar(255, 255, 255, 255);
    public Mat buildMask() {
        pathSearch();
        mask = Mat.ones(img1.size(), img1.type()).setTo(plusScalar);
        maskBuff = new byte[(int)img1.total() * channels];
        mask.get(0, 0, maskBuff);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                field[x][y].visited = false;
            }
        }

        bfsFill();
        mask.put(0, 0, maskBuff);

        return mask;
    }

    private void bfsFill() {
        Stack<Point> stack = new Stack<>();
        stack.push(field[0][0]);
        while (!stack.empty()) {
            Point p = stack.pop();
            setEmpty(p.x, p.y, maskBuff);
            for (Point neighbor : p.getNeighbors()) {
                if (!neighbor.visited && neighbor.type != PointType.PATH) {
                    neighbor.visited = true;
                    stack.push(neighbor);
                }
            }
        }
    }
}
