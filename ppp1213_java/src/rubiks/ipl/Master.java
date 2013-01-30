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
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;

public class Master implements MessageUpcall, ReceivePortConnectUpcall {

	private enum Status {
		INITIALIZING, FILLING_DEQUE, PROCESSING_DEQUE, DEQUE_EMPTY, WAITING_FOR_WORKERS, DONE
	}

	private final Rubiks parent;
	private final HashMap<IbisIdentifier, SendPort> senders;
	private ReceivePort receiver;
	private final Deque<Cube> deque;
	private Status status;
	private final AtomicInteger busyWorkers;
	private final AtomicInteger solutions;

	Master(Rubiks parent) throws IOException {
		this.parent = parent;
		senders = new HashMap<IbisIdentifier, SendPort>();
		deque = new ArrayDeque<Cube>();
		busyWorkers = new AtomicInteger(0);
		status = Status.INITIALIZING;
		solutions = new AtomicInteger(0);
	}

	/**
	 * Creates a receive port to receive cube requests from workers.
	 * @throws IOException
	 */
	private void openPorts() throws IOException {
		receiver = parent.ibis.createReceivePort(Rubiks.portType, "master",
				this, this, new Properties());
		receiver.enableConnections();
		receiver.enableMessageUpcalls();
	}

	/**
	 * Sends a termination message to all connected workers and closes all
	 * ports.
	 * @throws IOException
	 */
	public void shutdown() throws IOException {
		// Terminate the pool
		status = Status.DONE;
		parent.ibis.registry().terminate();

		// Close ports (and send termination messages)
		for (SendPort sender : senders.values()) {
			WriteMessage wm = sender.newMessage();
			wm.writeBoolean(true);
			wm.finish();
			sender.close();
		}
		receiver.close();
	}

	/**
	 * If a connection to the receive port is established, create a sendport in
	 * the reverse direction.
	 */
	@Override
	public boolean gotConnection(ReceivePort rp, SendPortIdentifier spi) {
		try {
			IbisIdentifier worker = spi.ibisIdentifier();
			SendPort sender = parent.ibis.createSendPort(Rubiks.portType);
			sender.connect(worker, "worker");
			senders.put(worker, sender);
		} catch (IOException e) {
			e.printStackTrace(System.err);
		}
		return true;
	}

	/**
	 * If a connection to the receive port is lost, close the reverse
	 * connection.
	 */
	@Override
	public void lostConnection(ReceivePort rp, SendPortIdentifier spi, Throwable thrwbl) {
		try {
			IbisIdentifier worker = spi.ibisIdentifier();
			SendPort sender = senders.get(worker);
			sender.close();
			senders.remove(worker);
		} catch (IOException e) {
			e.printStackTrace(System.err);
		}
	}

	/**
	 * Waits until all workers have finished their work and sent the number of
	 * solutions.
	 */
	private void waitForWorkers() throws InterruptedException {
		synchronized (this) {
			status = Status.WAITING_FOR_WORKERS;
			while (busyWorkers.get() != 0) {
				this.wait();
			}
		}
	}

	/**
	 * Get the last cube from the deque, which will have less twists and thus
	 * more work on average than cubes from the start of the deque.
	 */
	private Cube getLast() {
		Cube cube = null;
		try {
			synchronized (deque) {
				while (status != Status.PROCESSING_DEQUE) {
					deque.wait();
				}
				cube = deque.removeLast();
				if (deque.isEmpty()) {
					status = Status.DEQUE_EMPTY;
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace(System.err);
		}
		return cube;
	}

	/**
	 * Send a cube to a worker.
	 */
	void sendCube(Cube cube, IbisIdentifier destination) throws IOException {
		SendPort port = senders.get(destination);
		WriteMessage wm = port.newMessage();
		wm.writeBoolean(false);
		wm.writeObject(cube);
		wm.finish();
	}

	/**
	 * Processes a cube request / notification of found solutions from a worker.
	 */
	@Override
	public void upcall(ReadMessage rm) throws IOException, ClassNotFoundException {
		// Process the incoming message and decrease the number of busy workers
		IbisIdentifier sender = rm.origin().ibisIdentifier();
		int requestValue = rm.readInt();
		rm.finish();
		if (requestValue != Rubiks.DUMMY_VALUE) {
			synchronized (this) {
				solutions.addAndGet(requestValue);
				busyWorkers.decrementAndGet();
				this.notify();
			}
		}

		// Get the port to the sender and send the cube
		Cube replyValue = getLast(); // may block for some time
		sendCube(replyValue, sender);

		// Increase the number of workers we are waiting for
		busyWorkers.incrementAndGet();
	}

	/**
	 * Processes the first cube from the deque. This version is not recursive
	 * and slightly slower that the recursive one, but is easier to handle in
	 * the presence of other threads working on the deque.
	 *
	 * The
	 */
	private void processFirst(CubeCache cache) {
		synchronized (deque) {
			// Get a cube from the deque, null if deque is empty
			Cube cube = deque.pollFirst();
			if (cube == null) {
				status = Status.DEQUE_EMPTY;
				return;
			}

			// If the cube is solved, increment the number of found solutions
			if (cube.isSolved()) {
				solutions.incrementAndGet();
				cache.put(cube);
				return;
			}

			// Stop searching at the bound
			if (cube.getTwists() >= cube.getBound()) {
				cache.put(cube);
				return;
			}

			// Generate all possible cubes from this one by twisting it in
			// every possible way. Gets new objects from the cache
			Cube[] children = cube.generateChildren(cache);

			// Add all children to the beginning of the deque
			for (Cube child : children) {
				deque.addFirst(child);
			}
			
			// Make sure we have generated at least 12 * 12 = 144 children
			// before we let the workers steal jobs
			if (cube.getTwists() >= 2 && status == Status.FILLING_DEQUE) {
				status = Status.PROCESSING_DEQUE;
				deque.notifyAll();
			}
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
	private void solve(Cube cube) throws InterruptedException, IOException {
		// cache used for cube objects. Doing new Cube() for every move
		// overloads the garbage collector
		CubeCache cache = new CubeCache(cube.getSize());

		int bound = 0;
		System.out.print("Bound now:");

		while (solutions.get() == 0) {
			status = Status.FILLING_DEQUE;
			bound++;
			cube.setBound(bound);
			deque.addFirst(cube);

			System.out.print(" " + bound);
			while (!deque.isEmpty()) {
				processFirst(cache);
			}
			waitForWorkers();
		}
		shutdown();

		System.out.println();
		System.out.println("Solving cube possible in " + solutions + " ways of "
				+ bound + " steps");
	}

	private void printUsage() {
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
				parent.ibis.registry().terminate();
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
				parent.ibis.registry().terminate();
				System.exit(1);
			}
		}

		// print cube info
		System.out.println("Searching for solution for cube of size "
				+ cube.getSize() + ", twists = " + twists + ", seed = " + seed);
		cube.print(System.out);
		System.out.flush();

		// open Ibis ports
		openPorts();

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
