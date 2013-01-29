package rubiks.ipl;

import ibis.ipl.IbisIdentifier;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.RegistryEventHandler;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.EOFException;
import java.io.IOException;

public class Worker implements RegistryEventHandler {

	private final Rubiks parent;
	private CubeCache cache;
	private ReceivePort receiver;
	private SendPort sender;
	private boolean running;

	Worker(Rubiks parent) {
		this.parent = parent;
		this.cache = null;
		this.running = true;
	}

	void openPorts(IbisIdentifier master) throws IOException {
		// Create a reveive port and enable it
		receiver = parent.ibis.createReceivePort(Rubiks.explicitPortType, "worker");
		receiver.enableConnections();

		// Create a send port for sending requests and connect
		sender = parent.ibis.createSendPort(Rubiks.upcallPortType);
		sender.connect(master, "master");
	}
	
	void closePorts() throws IOException {
		sender.close();
		receiver.close();
	}
	
	void run(IbisIdentifier master) throws IOException, ClassNotFoundException {
		openPorts(master);

		// Send an initialization message
		sendValue(Rubiks.DUMMY_VALUE);
		
		// Send a message, receive a cube, solve it and reply number of solutions
		try {
			while (running) {
				Cube cube = receiveCube();
				if (cache == null) {
					cache = new CubeCache(cube.getSize());
				}
				sendValue(solutions(cube, cache));
			}
		} catch (EOFException e) {
			System.out.println(parent.ibis.identifier().toString() + " eof");
			// do nothing, occurs when master closes connection
		}
	}
	
	void sendValue(int value) throws IOException {
		WriteMessage wm = sender.newMessage();
		wm.writeInt(value);
		wm.finish();
	}
	
	Cube receiveCube() throws IOException, ClassNotFoundException {
		ReadMessage rm = receiver.receive();
		Cube result = (Cube) rm.readObject();
		rm.finish();
		return result;
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

	@Override
	public void joined(IbisIdentifier ii) {
	}

	@Override
	public void left(IbisIdentifier ii) {
	}

	@Override
	public void died(IbisIdentifier ii) {
	}

	@Override
	public void gotSignal(String string, IbisIdentifier ii) {
	}

	@Override
	public void electionResult(String string, IbisIdentifier ii) {
	}

	@Override
	public void poolClosed() {
		System.out.println(parent.ibis.identifier().toString() + " pool closed");
	}

	@Override
	public void poolTerminated(IbisIdentifier ii) {
		System.out.println(parent.ibis.identifier().toString() + " pool terminated");
		try {
			running = false;
			closePorts();
		} catch (IOException e) {
			// Nothing we can do, pool is terminated anyway
		}

	}
}
