import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class GameController implements ActionListener {

    public static int titleBarHeight;
    public static int windowTopOffset;
    public static Dimension windowDim;
    public static ArrayList<Point> magPosList;
    public static ArduinoSerialWriter serialWriter;
    public static boolean arduinoConnected;

    private boolean onStartScreen;
    private String username;

    private JFrame frame;
    private JPanel startPanel;
    private GamePanel gamePanel;
    private JButton startButton;
    private JTextField nameTextField;

    // --------------------------------- //
    // --- constructor and listeners --- //
    // --------------------------------- //
    public GameController() {
        // init JFrame stuff
        this.frame = new JFrame();
        this.frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        this.frame.setName("Haptic Feedback Test Environment");
        this.frame.setBackground(Color.DARK_GRAY);
        windowDim = GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds().getSize();
        this.frame.setPreferredSize(windowDim);
        
        // set up global serial writer
        GameController.serialWriter = new ArduinoSerialWriter();
        GameController.serialWriter.setupSerialComm();
        GameController.arduinoConnected = GameController.serialWriter.isArduinoConnected();

        // set up other global variables
        this.onStartScreen = false;
        this.username = "";
        titleBarHeight = this.frame.getY();
        windowTopOffset = this.frame.getInsets().top;
        magPosList = new ArrayList<Point>();
        initMagPositions();

        // set up start panel
        initStartPanel();
        this.frame.add(startPanel);
        this.frame.setResizable(false);
        this.frame.pack();
        this.frame.setLocationRelativeTo(null);
        this.frame.setVisible(true);

        // init listeners
        this.frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                restartPrompt();
            }
        });
    }

    // ------------------------- //
    // --- overriden methods --- //
    // ------------------------- //
    @Override
    public void actionPerformed(ActionEvent e) {
        openGamePanel(e);
    }

    // ----------------------------------- //
    // --- screen navigation and setup --- //
    // ----------------------------------- //
    private void initMagPositions() {
        int xSpacing = (int) windowDim.getWidth() / 10;
        int ySpacing = (int) (windowDim.getHeight()-(titleBarHeight + windowTopOffset)) / 6;
        int colStart;
        int colEnd;
        for (int row=1; row<6; row++) {
            colStart = (row % 2 == 0) ? 2 : 1;
            colEnd = (row % 2 == 0) ? 9 : 10;
            for (int col=colStart; col<colEnd; col+=2) {
                // this saves the position of the top-left corner of each magnet circle
                magPosList.add(new Point(col*xSpacing, row*ySpacing));
            }
        }
    }

    private void initStartPanel() {
        this.onStartScreen = true;
        
        this.startPanel = new JPanel();
        this.startPanel.setPreferredSize(windowDim);
        this.startPanel.setBackground(Color.LIGHT_GRAY);
        this.startPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 400, 125));
        
        JLabel titleLabel = new JLabel("Haptic Feedback Test Environment");
        titleLabel.setFont(new Font("Calibri", Font.BOLD, 50));

        this.startButton = new JButton("Press to Start");
        this.startButton.setPreferredSize(new Dimension(600, 200));
        this.startButton.setFont(new Font("Calibri", Font.PLAIN, 30));

        JPanel nameFieldPanel = new JPanel();
        JLabel nameFieldLabel = new JLabel("Name: ");
        nameFieldLabel.setFont(new Font("Lato", Font.PLAIN, 25));
        this.nameTextField = new JTextField(24);
        this.nameTextField.setPreferredSize(new Dimension(325, 50));
        this.nameTextField.setFont(new Font("Lato", Font.PLAIN, 20));
        nameFieldPanel.setBackground(Color.LIGHT_GRAY);
        nameFieldPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 20, 20));
        nameFieldPanel.setPreferredSize(new Dimension(800, 80));
        nameFieldPanel.add(nameFieldLabel);
        nameFieldPanel.add(this.nameTextField);

        this.startPanel.add(titleLabel);
        this.startPanel.add(nameFieldPanel);
        this.startPanel.add(startButton);

        this.startButton.addActionListener(this);
        this.nameTextField.addKeyListener(new KeyAdapter() {
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                    startButton.doClick();
                }
            }
        });
    }

    private void restartPrompt() {
        if (onStartScreen) {
            if (arduinoConnected)
                serialWriter.closeSerialComm();
            this.frame.dispose();
            System.exit(0);

        } else {
            Object options[] = { "Yes, exit", "Restart testing", "Cancel" };
            int restartDialogButton = JOptionPane.showOptionDialog(frame, "Are you sure you want to exit?", 
                "Select an Option", JOptionPane.YES_NO_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[0]);
            
            // restarting the app
            if (restartDialogButton == JOptionPane.NO_OPTION) {
                this.frame.getContentPane().remove(gamePanel);
                initStartPanel();
                this.frame.add(startPanel);
                this.frame.pack();
                this.frame.revalidate();
                this.frame.repaint();
    
            // closing the app
            } else if (restartDialogButton == JOptionPane.YES_OPTION) {
                if (arduinoConnected)
                    serialWriter.closeSerialComm();
                this.frame.dispose();
                System.exit(0);
            }
        }
    } 

    private void openGamePanel(ActionEvent e) {
        String command = e.getActionCommand();
        if (command.equals("Press to Start")) {
            String name = this.nameTextField.getText().toLowerCase();
            if (name.isBlank() || name.isEmpty()) {
                JOptionPane.showMessageDialog(null, "Please enter a name to proceed");
            } else {
                this.username = name;
                this.onStartScreen = false;
                this.gamePanel = new GamePanel(this.username);
                this.frame.getContentPane().remove(startPanel);
                this.frame.add(this.gamePanel);
                this.frame.pack();
                this.frame.revalidate();
                this.frame.repaint();
            }
        }
    }

}
