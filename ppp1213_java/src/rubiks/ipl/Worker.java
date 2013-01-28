package rubiks.ipl;

import ibis.ipl.IbisIdentifier;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.IOException;

public class Worker {

	private final Rubiks parent;
	private CubeCache cache;

	Worker(Rubiks parent) {
		this.parent = parent;
	}

	void run(IbisIdentifier master) throws IOException, ClassNotFoundException {
		// Create a reveive port and enable it
		ReceivePort receiver = parent.ibis.createReceivePort(Rubiks.explicitPortType, "worker");
		receiver.enableConnections();

		// Create a send port for sending requests and connect
		SendPort sender = parent.ibis.createSendPort(Rubiks.upcallPortType);
		sender.connect(master, "master");

		// Send a message, receive a cube, solve it and reply number of solutions
		Cube cube = null;
		boolean first = true;
		do {
			//try {
				int solutions;
				if (first) {
					solutions = Rubiks.DUMMY_VALUE;
				} else {
					solutions = solutions(cube, cache);
				}
				WriteMessage wm = sender.newMessage();
				wm.writeInt(solutions);
				wm.finish();
				ReadMessage rm = receiver.receive();
				cube = (Cube) rm.readObject();
				rm.finish();
				if (first && cube != null) {
					cache = new CubeCache(cube.getSize());
					first = false;
				}
			//} catch (IOException e) {
			//	break;
			//}
		} while (cube != null);
		
		// Close ports.
		sender.close();
		receiver.close();
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
}
