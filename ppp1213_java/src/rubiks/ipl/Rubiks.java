package rubiks.ipl;

import ibis.ipl.Ibis;
import ibis.ipl.IbisCapabilities;
import ibis.ipl.IbisCreationFailedException;
import ibis.ipl.IbisFactory;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.PortType;
import java.io.IOException;

/**
 * Solver for Rubik's cube puzzle
 */
public class Rubiks {
	
	public final PortType upcallPortType = new PortType(
			PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT_IBIS,
			PortType.RECEIVE_AUTO_UPCALLS, PortType.CONNECTION_ONE_TO_ONE);
	
	public final PortType explicitPortType = new PortType(
			PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT_IBIS,
			PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_ONE_TO_ONE);

    public final IbisCapabilities ibisCapabilities = new IbisCapabilities(
			IbisCapabilities.ELECTIONS_STRICT);

    public final Ibis ibis;
	
	Rubiks() throws IbisCreationFailedException {
		// Create an ibis instance
        ibis = IbisFactory.createIbis(ibisCapabilities, null,
				upcallPortType, explicitPortType);
	}
	
	void run(String[] arguments) throws IOException, ClassNotFoundException {
        // Elect a master
        IbisIdentifier master = ibis.registry().elect("Master");

        // If I am the master, run master, else run worker
        if (master.equals(ibis.identifier())) {
			System.out.println("Master: hello");
            new Master(this).run(arguments);
        } else {
			System.out.println("Worker: hello");
            new Worker(this).run(master);
        }

        // End ibis
        ibis.end();
	}
	
	public static void main(String[] arguments) {
		try {
            new Rubiks().run(arguments);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
	}
}
