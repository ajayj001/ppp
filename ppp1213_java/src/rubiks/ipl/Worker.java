package rubiks.ipl;

import ibis.ipl.IbisIdentifier;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.IOException;

public class Worker {
	
	private final Rubiks parent;
	
	Worker(Rubiks parent) {
		this.parent = parent;
	}
	
	void run(IbisIdentifier master) throws IOException, ClassNotFoundException {
		// Create a reveive port and enable it
		ReceivePort receiver = parent.ibis.createReceivePort(parent.explicitPortType, "worker");
		receiver.enableConnections();
		
		// Create a send port for sending requests and connect
        SendPort sender = parent.ibis.createSendPort(parent.upcallPortType);
        sender.connect(master, "master");

        // Send the message
        WriteMessage wm = sender.newMessage();
        wm.finish();

		// Receive a reply
		ReadMessage rm = receiver.receive();
		Cube cube = (Cube) rm.readObject();
		Solver.solve(cube);
		
        // Close ports.
        sender.close();
		receiver.close();
	}
}
