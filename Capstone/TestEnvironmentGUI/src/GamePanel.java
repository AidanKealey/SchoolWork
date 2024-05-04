import java.awt.AWTException;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import javax.swing.JPanel;
import javax.swing.Timer;

public class GamePanel extends JPanel implements ActionListener {

    private static Color CUSTOM_GREEN;

    private int totalInsets;
    private int currentRound;
    private int numClicks;
    private Timer timer;
    private Point mousePos;
    private Point targetPos;
    private Point guessPos;
    private String username;
    private boolean roundsComplete;
    private ArrayList<Integer> distList;
    private ArrayList<Double> guessTimeList;

    // -------------------------------- //
    // ----- constructor and init ----- //
    // -------------------------------- //
    public GamePanel(String username) {
        CUSTOM_GREEN = new Color(30, 201, 139);
        this.totalInsets = GameController.windowTopOffset + GameController.titleBarHeight;
        this.currentRound = 0;
        this.numClicks = 0;
        this.username = username;
        this.distList = new ArrayList<Integer>();
        this.guessTimeList = new ArrayList<Double>();
        
        // set up calibration
        startMouseCalibration();

        // set up JPanel stuff
        this.setPreferredSize(GameController.windowDim);
        this.setBackground(new Color(232, 232, 232));
        this.setFocusable(true);
        this.requestFocusInWindow();

        // set up tick functionality
        this.timer = new Timer(Consts.DELAY, this);
        this.timer.setActionCommand("tick");
        this.timer.start();

        // mouse clicks paint guesses on screen
        this.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                manageGameTraversal(e);
            }
        });
    }

    // ----------------------------- //
    // ----- game loop methods ----- //
    // ----------------------------- //
    private void startMouseCalibration() {
        if (GameController.arduinoConnected) {
            GameController.serialWriter.turnOnCoils(Consts.CALIBRATE_COILS);
        }
        numClicks = 0;
    }

    private void finishMouseCalibration() {
        if (GameController.arduinoConnected) {
            GameController.serialWriter.turnOnCoils(Consts.RESET_COILS);
        }
        setCursorInMiddle();
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void setCursorInMiddle() {
        try {
            Robot robot = new Robot();
            int centerX = GameController.magPosList.get(11).x;
            int centerY = GameController.magPosList.get(11).y + 65;
            robot.mouseMove(centerX, centerY);
        } catch (AWTException ex) {
            ex.printStackTrace();
        }
    }

    private void generateNewTarget() {
        currentRound++;
        // randomly create new target x and y
        int targetX = ThreadLocalRandom.current().nextInt(0+(2*Consts.TARGET_RADIUS), (int)GameController.windowDim.getWidth()-(2*Consts.TARGET_RADIUS)+1);
        int targetY = ThreadLocalRandom.current().nextInt(0+(2*Consts.TARGET_RADIUS), (int)GameController.windowDim.getHeight()-(2*Consts.TARGET_RADIUS)+1);
        this.targetPos = new Point(targetX, targetY);
        // send activation string to arduino
        String bitString = genCoilSerialString();
        if (GameController.arduinoConnected) {
            GameController.serialWriter.turnOnCoils(bitString);
        }
        // start new timing for a round
        this.guessTimeList.add((double) System.currentTimeMillis());
    }

    private void makeNewGuess(MouseEvent e) {;
        // record time taken for user to make guess
        double startTime = guessTimeList.get(guessTimeList.size() - 1);
        double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0;
        guessTimeList.set(guessTimeList.size() - 1, elapsedTime);
        // make new guess for current round
        guessPos = new Point(e.getX(), e.getY());
        distList.add(calcDistance(guessPos, targetPos));
        if (GameController.arduinoConnected) {
            GameController.serialWriter.turnOnCoils(Consts.RESET_COILS);
        }
    }

    private void finishGame() {
        if (GameController.arduinoConnected)
            GameController.serialWriter.turnOnCoils(Consts.RESET_COILS);
        if (Consts.SAVE_ENABLED)
            SaveUtil.saveToCsv(username, distList, guessTimeList);
        roundsComplete = true;
    }

    private int calcDistance(Point a, Point b) {
        return (int) Math.hypot(a.x - b.x, a.y - b.y);
    }

    private String genCoilSerialString() {
        StringBuilder bitString = new StringBuilder(23);
        for (Point p : GameController.magPosList) {
            int hyp = calcDistance(p, this.targetPos);
            int bit = (hyp < Consts.ACTIVATION_RADIUS) ? 1 : 0;
            bitString.append(bit);
        }
        System.out.println("Round "+this.currentRound+": "+bitString.toString());
        return bitString.toString();
    }

    private void manageGameTraversal(MouseEvent e) {
        if (!roundsComplete) {

            numClicks++;

            switch(numClicks) {
                case 1: // finish mouse calibration
                    finishMouseCalibration();
                    generateNewTarget();
                    break;
                case 2: // user makes a guess
                    makeNewGuess(e);
                    break;
                case 3: // user is ready to recalibrate or show final results
                    if (currentRound == Consts.MAX_ROUNDS)
                        finishGame();
                    else 
                        startMouseCalibration();
                    break;
                default: 
                    System.out.println("numClicks variable out of bounds for manageGameTraversal()");
                    break;
            }
        }
    }
   
    // ----------------------------- //
    // ----- overriden methods ----- //
    // ----------------------------- //
    @Override
    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        if (command.equals("tick")) {
            revalidate();
            repaint(); // will call paintComponent()
        }
    }
   
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        // draw graphics.
        if (roundsComplete) {
            drawResultsScreen(g);
        } else {
            switch(numClicks) {
                case 0: 
                    drawCalibrationScreen(g);
                    break;
                case 1: 
                    drawGuessScreen(g);
                    break;
                case 2: 
                    drawMagnetCircles(g);
                    drawMousePosLabel(g);
                    drawTargetCircle(g);
                    drawGuessCircle(g);
                    drawNextRoundInstructions(g);
                    break;
                default: 
                    System.out.println("numClicks variable out of bounds for paintComponent()");
                    break;
            }
        }

        // this smooths out animations on some systems
        Toolkit.getDefaultToolkit().sync();
    }
   
    // ----------------------- //
    // --- drawing methods --- //
    // ----------------------- //
    private void drawText(Graphics g, String text, Font f, Color c, int width, int y) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
        g2d.setColor(c);
        g2d.setFont(f);
        FontMetrics metrics = g2d.getFontMetrics(g2d.getFont());
        
        Rectangle rect = new Rectangle(0, y, width, 50);
        int xCoord = rect.x + (rect.width - metrics.stringWidth(text)) / 2;
        int yCoord = rect.y + ((rect.height - metrics.getHeight()) / 2) + metrics.getAscent();

        g2d.drawString(text, xCoord, yCoord);
    }
   
    private void drawMousePosLabel(Graphics g) {
        this.mousePos = MouseInfo.getPointerInfo().getLocation();
        int yPos = this.mousePos.y - totalInsets;
        if (this.mousePos.y < totalInsets) {
            yPos = 0;
        } else if (this.mousePos.y > GameController.windowDim.getHeight() + GameController.windowTopOffset) {
            yPos = (int) GameController.windowDim.getHeight();
        }
        String text = "x: "+this.mousePos.x+"   "+"y: "+yPos;
        Font f = new Font("Lato", Font.PLAIN, 25);
        drawText(g, text, f, Color.BLACK, (int) GameController.windowDim.getWidth()/6, 5);
    }

    private void drawGuessCircle(Graphics g) {
        g.setColor(Color.ORANGE);
        g.fillOval(guessPos.x-Consts.GUESS_RADIUS, guessPos.y-Consts.GUESS_RADIUS, 2*Consts.GUESS_RADIUS, 2*Consts.GUESS_RADIUS);
        g.setColor(Color.BLACK);
        g.drawLine(guessPos.x, guessPos.y, targetPos.x, targetPos.y);
        String text = "Round "+currentRound+" distance to target: "+calcDistance(guessPos, targetPos)+" pixels";
        Font f = new Font("Lato", Font.BOLD, 30);
        drawText(g, text, f, Color.BLACK, (int) GameController.windowDim.getWidth(), 5);
    }

    private void drawTargetCircle(Graphics g) {
        g.setColor(CUSTOM_GREEN);
        g.fillOval(targetPos.x-Consts.TARGET_RADIUS, targetPos.y-Consts.TARGET_RADIUS, 2*Consts.TARGET_RADIUS, 2*Consts.TARGET_RADIUS);
        g.setColor(Color.RED);
        g.drawOval(targetPos.x-Consts.ACTIVATION_RADIUS, targetPos.y-Consts.ACTIVATION_RADIUS, 2*Consts.ACTIVATION_RADIUS, 2*Consts.ACTIVATION_RADIUS);
    }

    private void drawMagnetCircles(Graphics g) {
        for (Point p : GameController.magPosList) {
            int hyp = calcDistance(p, this.targetPos);
            if (hyp < Consts.ACTIVATION_RADIUS) {
                g.setColor(Color.blue);
                g.fillOval(p.x-5, p.y-5, 10, 10);
            } else {
                g.setColor(Color.gray);
            }
            g.drawOval(p.x-Consts.MAGNET_RADIUS, p.y-Consts.MAGNET_RADIUS, Consts.MAGNET_RADIUS*2, Consts.MAGNET_RADIUS*2);
        }
    }

    private void drawGuessScreen(Graphics g) {
        // draw example circles in grey
        for (Point p : GameController.magPosList) {
            g.setColor(Color.gray);
            g.drawOval(p.x-Consts.MAGNET_RADIUS, p.y-Consts.MAGNET_RADIUS, Consts.MAGNET_RADIUS*2, Consts.MAGNET_RADIUS*2);
        }
        // draw text
        int yCoord = (int) (GameController.windowDim.getHeight()-totalInsets) / 4;
        Font f1 = new Font("Lato", Font.BOLD, 75);
        Font f2 = new Font("Lato", Font.BOLD, 40);
        String text1 = "Round "+this.currentRound;
        String text2 = "Click when you think you have found the target.";
        drawText(g, text1, f1, Color.BLACK, (int) GameController.windowDim.getWidth(), yCoord*1);
        drawText(g, text2, f2, Color.BLACK, (int) GameController.windowDim.getWidth(), yCoord*2);
    }

    private void drawNextRoundInstructions(Graphics g) {
        int yCoord = (int) (GameController.windowDim.getHeight()-totalInsets) - 60;
        Font f = new Font("Lato", Font.BOLD, 60);
        String text = "CLICK anywhere for next round...";
        drawText(g, text, f, Color.BLACK, (int) GameController.windowDim.getWidth(), yCoord);
    }

    private void drawResultsScreen(Graphics g) {
        DecimalFormat df = new DecimalFormat("#.##");
        int yScreen = (int) GameController.windowDim.getHeight()-totalInsets;
        int ySpacing = (int) yScreen / 10;
        drawText(g, "Results for \""+username+"\"", new Font("Lato", Font.BOLD, 60), Color.BLACK, (int) GameController.windowDim.getWidth(), 100);
        Font scoreFont = new Font("Lato", Font.PLAIN, 40);
        int sumScore = 0;
        double sumTime = 0l;
        for (int idx=0; idx<5; idx++) {
            sumScore += distList.get(idx);
            sumTime += guessTimeList.get(idx);
            String text = "Round "+(idx+1)+": "+distList.get(idx)+" pixels, "+df.format(guessTimeList.get(idx))+" seconds";
            drawText(g, text, scoreFont, Color.BLACK, (int) GameController.windowDim.getWidth(), (idx+3)*ySpacing);
        }
        int avgScore = (int) sumScore / 5;
        double avgTime = (double) sumTime / 5.0;
        Font avgFont = new Font("Lato", Font.BOLD, 50);
        drawText(g, "Average: "+avgScore+" pixels, "+df.format(avgTime)+" seconds", avgFont, Color.BLACK, (int) GameController.windowDim.getWidth(), yScreen-100);
    }

    private void drawCalibrationScreen(Graphics g) {
        // draw magnet location circles with highlited center
        for (int i=0; i<GameController.magPosList.size(); i++) {
            Point p = GameController.magPosList.get(i);
            if (i==11) {
                g.setColor(Color.blue);
                g.fillOval(p.x-5, p.y-5, 10, 10);
            } else {
                g.setColor(Color.gray);
            }
            g.drawOval(p.x-Consts.MAGNET_RADIUS, p.y-Consts.MAGNET_RADIUS, Consts.MAGNET_RADIUS*2, Consts.MAGNET_RADIUS*2);
        }
        // draw title
        Font f = new Font("Lato", Font.PLAIN, 30);
        String title = "Place the mouse in the center of the mouse pad and CLICK";
        drawText(g, title, f, Color.BLACK, (int) GameController.windowDim.getWidth(), 5);
    }

    
}
