package rubiks.ipl;

import ibis.ipl.IbisIdentifier;
import ibis.ipl.MessageUpcall;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.ReceivePortConnectUpcall;
import ibis.ipl.SendPort;
import ibis.ipl.SendPortIdentifier;
import ibis.ipl.WriteMessage;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;

public class Master implements MessageUpcall, ReceivePortConnectUpcall {

	private enum Status {
		INITIALIZING, FILLING_DEQUE, PROCESSING_DEQUE, DEQUE_EMPTY, WAITING_FOR_WORKERS, DONE
	}
	
	private final Rubiks parent;
	private final HashMap<IbisIdentifier, SendPort> sendports;
	private final BlockingDeque<Cube> deque;
	private final Object lock;
	private Status status;
	private final AtomicInteger busyWorkers;
	private final AtomicInteger solutions;

	Master(Rubiks parent) throws IOException {
		this.parent = parent;
		this.sendports = new HashMap<IbisIdentifier, SendPort>();
		this.deque = new LinkedBlockingDeque<Cube>();
		this.busyWorkers = new AtomicInteger(0);
		this.status = Status.INITIALIZING;
		System.out.println("status: initializing");
		this.lock = new Object();
		this.solutions = new AtomicInteger(0);
	}

	void configure() throws IOException {
		// Create a receive port and enable it
		ReceivePort receiver = parent.ibis.createReceivePort(Rubiks.upcallPortType, "master", this, this, new Properties());
		receiver.enableConnections();
		receiver.enableMessageUpcalls();
	}
	
	@Override
	public boolean gotConnection(ReceivePort rp, SendPortIdentifier spi) {
		try {
			IbisIdentifier worker = spi.ibisIdentifier();
			SendPort sender = parent.ibis.createSendPort(Rubiks.explicitPortType);
			sender.connect(worker, "worker");
			sendports.put(worker, sender);
		} catch (IOException ex) {
		}
		return true;
	}

	@Override
	public void lostConnection(ReceivePort rp, SendPortIdentifier spi, Throwable thrwbl) {
		try {
			IbisIdentifier worker = spi.ibisIdentifier();
			SendPort sender = sendports.get(worker);
			sender.close();
			sendports.remove(worker);
		} catch (IOException ex) {
		}
	}

	void waitForWorkers() throws InterruptedException {
		synchronized (lock) {
			status = Status.WAITING_FOR_WORKERS;
			System.out.println("status: waiting for workers");
			while (busyWorkers.get() != 0) {
				lock.wait();
			}
		}
	}

	Cube getLast() {
		Cube cube = null;
		try {
			synchronized (deque) {
				while (status != Status.PROCESSING_DEQUE || status == Status.DONE) {
					deque.wait();
				}
				if (status == Status.DONE) {
					cube = null;
				} else {
					cube = deque.takeLast();
				}
				if (deque.isEmpty()) {
					synchronized (lock) {
						status = Status.DEQUE_EMPTY;
					}
					System.out.println("status: deque empty");
				}
			}
		} catch (InterruptedException ex) {
		}
		return cube;
	}
	
	@Override
	public void upcall(ReadMessage rm) throws IOException, ClassNotFoundException {
		// Process the incoming message
		IbisIdentifier sender = rm.origin().ibisIdentifier();
		int requestValue = rm.readInt();
		rm.finish();
		synchronized (lock) {
			if (requestValue != Rubiks.DUMMY_VALUE) {
				solutions.addAndGet(requestValue);
				busyWorkers.decrementAndGet();
				System.out.println("Busyworkers: " + busyWorkers);
				lock.notifyAll();
			}
		}
		System.out.println("Master: got request with value " + requestValue);

		// Get the port to the sender and send a message
		SendPort port = sendports.get(sender);
		WriteMessage wm = port.newMessage();
		Cube replyValue = getLast(); // may block for some time
		wm.writeObject(replyValue);
		wm.finish();
		
		// Increase or decrease the number of workers we are waiting for
		if (replyValue != null) {
			busyWorkers.incrementAndGet();
			System.out.println("Busyworkers: " + busyWorkers);
			System.out.println("Master: sent a cube with " + replyValue.getTwists() + " twists");
		}
	}

	/**
	 * Recursive function to find a solution for a given cube. Only searches to
	 * the bound set in the cube object.
	 * 
	 * @param cube
	 *			cube to solve
	 * @param cache
	 *			cache of cubes used for new cube objects
	 * @return the number of solutions found
	 */
	void processFirst(CubeCache cache) {
		Cube cube = deque.pollFirst();
		if (cube == null) {
			status = Status.DEQUE_EMPTY;
			System.out.println("status: deque empty");
			System.out.println("Master: Got a null from the deque");
			return;
		}
		
		if (cube.isSolved()) {
			solutions.incrementAndGet();
			cache.put(cube);
			return;
		}

		if (cube.getTwists() >= cube.getBound()) {
			cache.put(cube);
			return;
		}
		
		synchronized (deque) {
			if (cube.getTwists() > 0 && status == Status.FILLING_DEQUE) {
				status = Status.PROCESSING_DEQUE;
				System.out.println("status: processing deque");
				deque.notifyAll();
			}
		}

		// generate all possible cubes from this one by twisting it in
		// every possible way. Gets new objects from the cache
		Cube[] children = cube.generateChildren(cache);

		for (Cube child : children) {
			deque.addFirst(child);
		}
	}

	/**
	 * Solves a Rubik's cube by iteratively searching for solutions with a
	 * greater depth. This guarantees the optimal solution is found. Repeats all
	 * work for the previous iteration each iteration though...
	 * 
	 * @param cube
	 *			the cube to solve
	 */
	void solve(Cube cube) throws InterruptedException {
		// cache used for cube objects. Doing new Cube() for every move
		// overloads the garbage collector
		CubeCache cache = new CubeCache(cube.getSize());

		int bound = 0;
		System.out.print("Bound now:");

		while (solutions.get() == 0) {
			status = Status.FILLING_DEQUE;
			System.out.println("status: filling deque");
			bound++;
			cube.setBound(bound);
			deque.addFirst(cube);

			System.out.print(" " + bound);
			while (!deque.isEmpty()) {
				processFirst(cache);
			}
			waitForWorkers();
		}
		status = Status.DONE;
		System.out.println("status: done");

		System.out.println();
		System.out.println("Solving cube possible in " + solutions + " ways of "
				+ bound + " steps");
	}

	void printUsage() {
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
	 *			list of arguments
	 */
	public void run(String[] arguments) throws IOException, InterruptedException {

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
				parent.ibis.end();
				System.exit(1);
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
		
		// configure Ibis ports
		configure();

		// solve
		long start = System.currentTimeMillis();
		solve(cube);
		long end = System.currentTimeMillis();

		// NOTE: this is printed to standard error! The rest of the output is
		// constant for each set of parameters. Printing this to standard error
		// makes the output of standard out comparable with "diff"
		System.err.println("Solving cube took " + (end - start)
				+ " milliseconds");
	}
}
