package rubiks.ipl;

import ibis.ipl.ConnectionClosedException;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.MessageUpcall;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.IOException;
import rubiks.sequential.Cube;
import rubiks.sequential.CubeCache;

public class Worker implements MessageUpcall {

	private final Rubiks parent;
	private CubeCache cache;
	private ReceivePort receiver;
	private SendPort sender;

	Worker(Rubiks parent) {
		this.parent = parent;
		cache = null;
	}

	void openPorts(IbisIdentifier master) throws IOException {
		receiver = parent.ibis.createReceivePort(Rubiks.portType, "worker", this);
		receiver.enableConnections();
		receiver.enableMessageUpcalls();

		sender = parent.ibis.createSendPort(Rubiks.portType);
		sender.connect(master, "master");
	}

	public void shutdown() throws IOException {
		// Close the ports
		try {
			sender.close();
			receiver.close();
		} catch (ConnectionClosedException e) {
			// do nothing
		}

		// Notify the main thread
		synchronized (this) {
			this.notify();
		}
	}

	void run(IbisIdentifier master) throws IOException, ClassNotFoundException, InterruptedException {
		// Open send and receive ports
		openPorts(master);

		// Send an initialization message
		sendInt(Rubiks.INIT_VALUE);

		// Make sure this thread doesn't finish prematurely
		synchronized (this) {
			this.wait();
		}
	}

	@Override
	public void upcall(ReadMessage rm) throws IOException, ClassNotFoundException {
		// Check whether we should terminate or not
		boolean shouldClose = rm.readBoolean();
		if (shouldClose) {
			rm.finish();
			shutdown();
		} else {
			// Process the cube and send back the number of solutions
			Cube cube = (Cube) rm.readObject();
			rm.finish();
			if (cache == null) {
				cache = new CubeCache(cube.getSize());
			}
			sendInt(solutions(cube, cache));
		}
	}

	void sendInt(int value) throws IOException {
		WriteMessage wm = sender.newMessage();
		wm.writeInt(value);
		wm.finish();
	}

	/**
	 * Recursive function to find a solution for a given cube. Only searches to
	 * the bound set in the cube object.
	 *
	 * @param cube cube to solve
	 * @param cache cache of cubes used for new cube objects
	 * @return the number of solutions found
	 */
	public int solutions(Cube cube, CubeCache cache) {
		if (cube.isSolved()) {
			return 1;
		}

		if (cube.getTwists() >= cube.getBound()) {
			return 0;
		}

		// generate all possible cubes from this one by twisting it in
		// every possible way. Gets new objects from the cache
		Cube[] children = cube.generateChildren(cache);

		int result = 0;

		for (Cube child : children) {
			// recursion step
			int childSolutions = solutions(child, cache);
			if (childSolutions > 0) {
				result += childSolutions;
			}
			// put child object in cache
			cache.put(child);
		}

		return result;
	}
}
