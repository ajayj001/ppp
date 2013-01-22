package rubiks.ipl;

import ibis.ipl.IbisIdentifier;
import ibis.ipl.MessageUpcall;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.IOException;
import java.util.HashMap;

public class Master implements MessageUpcall {

	private final Rubiks parent;
	private final SendPortMap sendports;
	
	Master(Rubiks parent) throws IOException {
		this.parent = parent;
		this.sendports = new SendPortMap();
	}
	
	class SendPortMap extends HashMap<IbisIdentifier, SendPort> {
		SendPort get(IbisIdentifier destination) throws IOException {
			SendPort result = super.get(destination);
			if (result == null) {
				System.out.println("Master: result null");
				result = parent.ibis.createSendPort(parent.explicitPortType);
				System.out.println("Master: created new port to " + destination);
				result.connect(destination, "worker");
				System.out.println("Master: port connected");
				sendports.put(destination, result);
				System.out.println("Master: saved sendport");
			}
			return result;
		}
	}
	
	void configure() throws IOException {
		// Create a receive port and enable it
        ReceivePort receiver = parent.ibis.createReceivePort(parent.upcallPortType, "master", this);
        receiver.enableConnections();
        receiver.enableMessageUpcalls();
	}
	
	@Override
	public void upcall(ReadMessage rm) throws IOException, ClassNotFoundException {
		IbisIdentifier sender = rm.origin().ibisIdentifier();
		rm.finish();
		SendPort port = sendports.get(sender);
		WriteMessage wm = port.newMessage();
		wm.writeObject(new Cube(3, 11, 0));
		wm.finish();
	}

    public void printUsage() {
        System.out.println("Rubiks Cube solver");
        System.out.println();
        System.out.println("Does a number of random twists, then solves the rubiks cube with a simple");
        System.out.println(" brute-force approach. Can also take a file as input");
        System.out.println();
        System.out.println("USAGE: Rubiks [OPTIONS]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("--size SIZE\t\tSize of cube (default: 3)");
        System.out.println("--twists TWISTS\t\tNumber of random twists (default: 11)");
        System.out.println("--seed SEED\t\tSeed of random generator (default: 0");
        System.out.println("--threads THREADS\t\tNumber of threads to use (default: 1, other values not supported by sequential version)");
        System.out.println();
        System.out.println("--file FILE_NAME\t\tLoad cube from given file instead of generating it");
        System.out.println();
    }

    /**
     * Main function.
     * 
     * @param arguments
     *            list of arguments
     */
    public void run(String[] arguments) throws IOException {
		
		// configure Ibis ports
		configure();
		
        Cube cube = null;

        // default parameters of puzzle
        int size = 3;
        int twists = 11;
        int seed = 0;
        String fileName = null;

        // number of threads used to solve puzzle
        // (not used in sequential version)

        for (int i = 0; i < arguments.length; i++) {
            if (arguments[i].equalsIgnoreCase("--size")) {
                i++;
                size = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--twists")) {
                i++;
                twists = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--seed")) {
                i++;
                seed = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--file")) {
                i++;
                fileName = arguments[i];
            } else if (arguments[i].equalsIgnoreCase("--help") || arguments[i].equalsIgnoreCase("-h")) {
                printUsage();
                System.exit(0);
            } else {
                System.err.println("unknown option : " + arguments[i]);
                printUsage();
                System.exit(1); //TODO shut down workers
            }
        }

        // create cube
        if (fileName == null) {
            cube = new Cube(size, twists, seed);
        } else {
            try {
                cube = new Cube(fileName);
            } catch (Exception e) {
                System.err.println("Cannot load cube from file: " + e);
                System.exit(1);
            }
        }
        
        // print cube info
        System.out.println("Searching for solution for cube of size "
                + cube.getSize() + ", twists = " + twists + ", seed = " + seed);
        cube.print(System.out);
        System.out.flush();

        
        // solve
        long start = System.currentTimeMillis();
        Solver.solve(cube);
        long end = System.currentTimeMillis();

        // NOTE: this is printed to standard error! The rest of the output is
        // constant for each set of parameters. Printing this to standard error
        // makes the output of standard out comparable with "diff"
        System.err.println("Solving cube took " + (end - start)
                + " milliseconds");
    }
}
