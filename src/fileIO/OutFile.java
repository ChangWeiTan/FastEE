/* Copyright (C) 2018 Chang Wei Tan, Francois Petitjean, Geoff Webb
 This file is part of FastEE.
 FastEE is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3 of the License.
 LbEnhanced is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>. */
package fileIO;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Original code from http://www.timeseriesclassification.com/code.php
 * https://bitbucket.org/TonyBagnall/time-series-classification/src/f47c300b937af1d06aac86bdbe228cb9a9d5e00d?at=default
 */
public class OutFile {

    private PrintWriter outFile;
    private char delimit;

    public OutFile(String name) {
        try {
            FileWriter fw = new FileWriter(name);
            outFile = new PrintWriter(fw);
            delimit = ',';
        } catch (IOException exception) {
            System.err.println(exception + " File " + name + " Not found");
        }
    }

    public OutFile(String name, boolean append) {
        try {
            FileWriter fw = new FileWriter(name, append);
            outFile = new PrintWriter(fw);
            delimit = ',';
        } catch (IOException exception) {
            System.err.println(exception + " File " + name + " Not found");
        }
    }

    public boolean writeString(String v) {
        outFile.print(v);
        return !outFile.checkError();
    }

    public boolean writeLine(String v) {
        outFile.print(v + "\n");
        return !outFile.checkError();
    }

    public boolean writeInt(int v) {
        outFile.print("" + v + delimit);
        return !outFile.checkError();
    }

    public boolean writeChar(char c) {
        outFile.print(c);
        return !outFile.checkError();
    }

    public boolean writeBoolean(boolean b) {
        outFile.print(b);
        return !outFile.checkError();
    }

    public boolean writeDouble(double v) {
        outFile.print("" + v + delimit);
        return !outFile.checkError();
    }

    public boolean newLine() {
        outFile.print("\n");
        return !outFile.checkError();
    }

    public void closeFile() {
        outFile.close();
    }
}
	