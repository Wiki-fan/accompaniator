package accompaniator_team.playwithme;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.SystemClock;
import android.support.annotation.Nullable;
import android.util.Log;

import org.billthefarmer.mididriver.MidiDriver;

import java.util.concurrent.LinkedBlockingQueue;

public class PlayerService extends Service {
    private static final String TAG = "PlayerService";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "onStartCommand");
        queueOut = (LinkedBlockingQueue<Chord>) intent.getSerializableExtra(QUEUE_NAME);

        PlayerService.Note[] notes = {new PlayerService.Note(100)};
        queueOut.add(new PlayerService.Chord(notes, 127, 127));

        midiDriver = new MidiDriver();
        mThread = new PlayerThread(queueOut, midiDriver);
        mThread.start();
        return super.onStartCommand(intent, flags, startId);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    public static class Note {
        int number;

        public Note(int number) {
            this.number = number;
        }
    }

    public static class Chord {
        Note[] notes;
        int velocity;
        int duration;

        public Chord(Note[] notes, int velocity, int duration) {
            this.notes = notes;
            this.velocity = velocity;
            this.duration = duration;
        }
    }

    public static final String QUEUE_NAME = "playQueue";
    LinkedBlockingQueue<Chord> queueOut;
    private MidiDriver midiDriver;
    private PlayerThread mThread;

    @Override
    public void onCreate() {
        Log.d(TAG, "onCreate");
        super.onCreate();
    }

    /*@Override
    protected void onResume() {
        if (midiDriver != null)
            midiDriver.start();
    }

    @Override
    protected void onPause() {
        if (midiDriver != null)
            midiDriver.stop();
    }*/

    @Override
    public void onDestroy() {
        midiDriver.stop();
        //mThread.join();
    }
}